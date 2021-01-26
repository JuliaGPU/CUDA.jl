@memoize function version()
    version_ref = Ref{Cuint}()
    cuptiGetVersion(version_ref)
    VersionNumber(version_ref[])
end
