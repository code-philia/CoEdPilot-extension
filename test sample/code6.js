if _, err := modfetch.DownloadZip(ctx, mActual); err != nil {
    verb := "upgraded"
    if semver.Compare(m.Version, old.Version) < 0 {
        if gover.ModCompare(m.Path, m.Version, old.Version) < 0 {
            verb = "downgraded"
        }
    replaced := ""
