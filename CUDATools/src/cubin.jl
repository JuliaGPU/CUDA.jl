# decoding the source locations of SASS instructions from a cubin's line table

# a source location, in the function `func` (empty if not recorded)
struct SourceFrame
    func::String
    file::String
    line::Int
end

# the source locations of the instructions in a function, as rows that apply from their
# offset to the next row's: the frames of every (outermost first), or none if unknown
struct LineTable
    offsets::Vector{Int}
    chains::Vector{Vector{SourceFrame}}
end

function source_chain(table::LineTable, offset::Integer)
    i = searchsortedlast(table.offsets, offset)
    i == 0 ? SourceFrame[] : table.chains[i]
end

# add a row to a line table that is being built; when rows share an offset, the last applies
function add_row!(offsets::Vector{Int}, chains::Vector{Vector{SourceFrame}}, offset::Int,
                  chain::Vector{SourceFrame})
    if !isempty(offsets) && offsets[end] == offset
        chains[end] = chain
    else
        push!(offsets, offset)
        push!(chains, chain)
    end
end


## ELF

struct ELFSection
    name::String
    type::UInt32
    offset::Int
    size::Int
    link::Int
    info::Int
end

struct ELFFile
    data::Vector{UInt8}
    sections::Vector{ELFSection}
end

function read_scalar(::Type{T}, data::Vector{UInt8}, offset::Integer) where {T}
    offset + sizeof(T) <= length(data) || throw(EOFError())
    GC.@preserve data unsafe_load(Ptr{T}(pointer(data, offset + 1)))
end

function read_cstring(data::Vector{UInt8}, offset::Integer)
    stop = findnext(iszero, data, offset + 1)
    stop === nothing && throw(EOFError())
    String(data[offset+1:stop-1])
end

function ELFFile(data::Vector{UInt8})
    # cubins are 64-bit little-endian ELF files
    length(data) >= 64 && data[1:4] == b"\x7fELF" && data[5] == 2 && data[6] == 1 ||
        error("Not a 64-bit little-endian ELF file")
    shoff = read_scalar(UInt64, data, 0x28)
    shentsize = read_scalar(UInt16, data, 0x3a)
    shnum = read_scalar(UInt16, data, 0x3c)
    shstrndx = read_scalar(UInt16, data, 0x3e)
    headers = map(0:shnum-1) do i
        header = shoff + i * shentsize
        (; name=read_scalar(UInt32, data, header), type=read_scalar(UInt32, data, header + 0x04),
           offset=Int(read_scalar(UInt64, data, header + 0x18)),
           size=Int(read_scalar(UInt64, data, header + 0x20)),
           link=Int(read_scalar(UInt32, data, header + 0x28)),
           info=Int(read_scalar(UInt32, data, header + 0x2c)))
    end
    names = headers[shstrndx+1].offset
    sections = ELFSection[ELFSection(read_cstring(data, names + h.name), h.type, h.offset,
                                     h.size, h.link, h.info) for h in headers]
    ELFFile(data, sections)
end

find_section(elf::ELFFile, name::String) = findfirst(s -> s.name == name, elf.sections)

section_bytes(elf::ELFFile, i::Int) =
    view(elf.data, elf.sections[i].offset+1:elf.sections[i].offset+elf.sections[i].size)

const SHT_SYMTAB = 2
const SHT_RELA = 4
const SHT_REL = 9

# the index of the symbol called `name`, or `nothing`
function symbol_index(elf::ELFFile, name::String)
    i = findfirst(s -> s.type == SHT_SYMTAB, elf.sections)
    i === nothing && return nothing
    symtab = elf.sections[i]
    strtab = elf.sections[symtab.link+1]
    for k in 0:div(symtab.size, 24)-1
        offset = read_scalar(UInt32, elf.data, symtab.offset + 24k)
        read_cstring(elf.data, strtab.offset + offset) == name && return k
    end
    return nothing
end

# the section index (0-based) every symbol is defined in
function symbol_sections(elf::ELFFile)
    i = findfirst(s -> s.type == SHT_SYMTAB, elf.sections)
    i === nothing && return Int[]
    symtab = elf.sections[i]
    Int[read_scalar(UInt16, elf.data, symtab.offset + 24k + 6) for k in 0:div(symtab.size, 24)-1]
end

# the symbol index of the relocations that apply to section `i`, by offset in that section.
# ptxas emits relocations with explicit addends for sm_90 and newer, and without for older
# architectures; we only need the symbol, as the address operand is relative to it anyway.
function section_relocations(elf::ELFFile, i::Int)
    relocations = Dict{Int,Int}()
    for section in elf.sections
        section.type in (SHT_REL, SHT_RELA) && section.info == i - 1 || continue
        entry_size = section.type == SHT_RELA ? 24 : 16
        for k in 0:div(section.size, entry_size)-1
            entry = section.offset + entry_size * k
            offset = read_scalar(UInt64, elf.data, entry)
            relocations[offset] = read_scalar(UInt64, elf.data, entry + 8) >> 32
        end
    end
    return relocations
end


## DWARF line table

function read_uleb128(data, pos::Int)
    result = UInt64(0)
    shift = 0
    while true
        byte = data[pos]
        pos += 1
        shift < 64 && (result |= UInt64(byte & 0x7f) << shift)
        shift += 7
        byte & 0x80 == 0 && return result, pos
    end
end

function read_sleb128(data, pos::Int)
    result = Int64(0)
    shift = 0
    byte = 0x00
    while true
        byte = data[pos]
        pos += 1
        shift < 64 && (result |= Int64(byte & 0x7f) << shift)
        shift += 7
        byte & 0x80 == 0 && break
    end
    if shift < 64 && byte & 0x40 != 0
        result |= -(Int64(1) << shift)
    end
    return result, pos
end

# a row of the line table. ptxas records the frames of inlined code with an extended
# opcode, which sets the (1-based) row describing the location the code was inlined at,
# and the offset in .debug_str of the name of the inlined function.
struct LineRow
    address::Int
    file::Int
    line::Int
    context::Int
    func::Int
end

const DW_LNE_end_sequence = 0x01
const DW_LNE_set_address = 0x02
const DW_LNE_NV_set_context = 0x90

# decode the line number programs in `data`, returning their file names and their rows per
# sequence, along with the symbol each sequence's address is relative to and the address
# where the sequence ends
function decode_line_programs(data::AbstractVector{UInt8}, relocations::Dict{Int,Int})
    units = []
    unit = 1
    while unit + 4 <= length(data)
        unit_length = Int(ltoh(reinterpret(UInt32, data[unit:unit+3])[1]))
        unit_length < 0xfffffff0 || error("64-bit DWARF is not supported")
        unit_end = unit + 4 + unit_length
        version = ltoh(reinterpret(UInt16, data[unit+4:unit+5])[1])
        2 <= version <= 4 || error("DWARF version $version is not supported")
        header_length = Int(ltoh(reinterpret(UInt32, data[unit+6:unit+9])[1]))
        pos = unit + 10
        program = pos + header_length
        min_instruction_length = Int(data[pos])
        pos += version >= 4 ? 3 : 2     # skip max_ops_per_inst and default_is_stmt
        line_base = Int(reinterpret(Int8, data[pos]))
        line_range = Int(data[pos+1])
        opcode_base = Int(data[pos+2])
        standard_lengths = data[pos+3:pos+opcode_base+1]
        pos += 2 + opcode_base

        directories = String[]
        while data[pos] != 0
            stop = findnext(iszero, data, pos)
            push!(directories, String(data[pos:stop-1]))
            pos = stop + 1
        end
        pos += 1
        files = String[]
        while data[pos] != 0
            stop = findnext(iszero, data, pos)
            name = String(data[pos:stop-1])
            pos = stop + 1
            directory, pos = read_uleb128(data, pos)
            _, pos = read_uleb128(data, pos)    # modification time
            _, pos = read_uleb128(data, pos)    # length
            if directory != 0 && !isabspath(name) && directory <= length(directories)
                name = joinpath(directories[directory], name)
            end
            push!(files, name)
        end

        rows = LineRow[]
        sequences = Tuple{Int,UnitRange{Int},Int}[]    # (symbol, rows, end address)
        address, file, line, context, func = 0, 1, 1, 0, 0
        symbol, first_row = -1, 1
        pos = program
        while pos < unit_end
            opcode = Int(data[pos])
            pos += 1
            if opcode >= opcode_base         # special opcode
                adjusted = opcode - opcode_base
                address += div(adjusted, line_range) * min_instruction_length
                line += line_base + mod(adjusted, line_range)
                push!(rows, LineRow(address, file, line, context, func))
            elseif opcode == 0              # extended opcode
                len, pos = read_uleb128(data, pos)
                start = pos
                len == 0 && continue
                sub = data[pos]
                if sub == DW_LNE_end_sequence
                    push!(sequences, (symbol, first_row:length(rows), address))
                    address, file, line, context, func = 0, 1, 1, 0, 0
                    symbol, first_row = -1, length(rows) + 1
                elseif sub == DW_LNE_set_address
                    symbol = get(relocations, start + 1, -1)
                    address = Int(ltoh(reinterpret(UInt64, data[pos+1:pos+8])[1]))
                elseif sub == DW_LNE_NV_set_context
                    c, p = read_uleb128(data, pos + 1)
                    f, _ = read_uleb128(data, p)
                    context, func = Int(c), Int(f)
                end
                pos = start + Int(len)
            elseif opcode == 1              # DW_LNS_copy
                push!(rows, LineRow(address, file, line, context, func))
            elseif opcode == 2              # DW_LNS_advance_pc
                delta, pos = read_uleb128(data, pos)
                address += Int(delta) * min_instruction_length
            elseif opcode == 3              # DW_LNS_advance_line
                delta, pos = read_sleb128(data, pos)
                line += Int(delta)
            elseif opcode == 4              # DW_LNS_set_file
                f, pos = read_uleb128(data, pos)
                file = Int(f)
            elseif opcode == 8              # DW_LNS_const_add_pc
                address += div(255 - opcode_base, line_range) * min_instruction_length
            elseif opcode == 9              # DW_LNS_fixed_advance_pc
                address += Int(ltoh(reinterpret(UInt16, data[pos:pos+1])[1]))
                pos += 2
            else                            # skip the operands of other standard opcodes
                for _ in 1:standard_lengths[opcode]
                    _, pos = read_uleb128(data, pos)
                end
            end
        end
        push!(units, (; files, rows, sequences))
        unit = unit_end
    end
    return units
end

"""
    line_tables(cubin::Vector{UInt8}) -> Dict{String,LineTable}

Decode the source locations of the instructions of every function in `cubin`, including
the frames of inlined functions, keyed by the function's (mangled) name. Functions without
line information are omitted.
"""
function line_tables(cubin::Vector{UInt8})
    elf = ELFFile(cubin)
    tables = Dict{String,LineTable}()
    i = find_section(elf, ".debug_line")
    i === nothing && return tables
    str = find_section(elf, ".debug_str")
    function_name(offset) = str === nothing ? "" :
        # Julia's function names end with a semicolon
        chopsuffix(read_cstring(elf.data, elf.sections[str].offset + offset), ";")
    sections = symbol_sections(elf)

    # relocations are keyed by their offset in the section; we look them up by the
    # (1-based) position of the address operand in the section data
    relocations = Dict(offset + 1 => symbol
                       for (offset, symbol) in section_relocations(elf, i))

    # a function's code can be covered by several sequences, each with its own rows
    sequences = Dict{String,Vector{LineTable}}()
    for unit in decode_line_programs(section_bytes(elf, i), relocations)
        file_name(i) = 1 <= i <= length(unit.files) ? unit.files[i] : ""
        frame(row) = SourceFrame(row.context == 0 ? "" : function_name(row.func),
                                 file_name(row.file), row.line)
        for (symbol, range, stop) in unit.sequences
            isempty(range) && continue
            1 <= symbol + 1 <= length(sections) || continue
            section = sections[symbol+1] + 1
            1 <= section <= length(elf.sections) || continue
            name = chopprefix(elf.sections[section].name, ".text.")
            name == elf.sections[section].name && continue

            offsets = Int[]
            chains = Vector{SourceFrame}[]
            for k in range
                address = unit.rows[k].address
                # follow the rows describing where code was inlined, which precede it
                row = unit.rows[k]
                chain = SourceFrame[frame(row)]
                parent = k
                while 0 < row.context < parent
                    parent = row.context
                    row = unit.rows[parent]
                    # a frame that is inlined at its own location (like Julia's `macro
                    # expansion` frames) refers to the previous row with that location,
                    # i.e., to itself in unrolled code. skip those repeated frames.
                    frame(row) == chain[end] && continue
                    push!(chain, frame(row))
                end
                # drop frames without a line, which some code generators emit for their
                # own code (e.g. LLVM.jl's generated functions)
                filter!(frame -> frame.line != 0, chain)
                reverse!(chain)
                add_row!(offsets, chains, address, chain)
            end
            # the location of code past the end of the sequence is unknown
            add_row!(offsets, chains, stop, SourceFrame[])
            push!(get!(Vector{LineTable}, sequences, name), LineTable(offsets, chains))
        end
    end

    for (name, parts) in sequences
        sort!(parts; by=part -> first(part.offsets))
        offsets = Int[]
        chains = Vector{SourceFrame}[]
        # a sequence that starts where the previous one ends replaces its end row
        for part in parts, (offset, chain) in zip(part.offsets, part.chains)
            add_row!(offsets, chains, offset, chain)
        end
        tables[name] = LineTable(offsets, chains)
    end
    return tables
end
