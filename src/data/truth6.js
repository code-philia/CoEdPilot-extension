if err != nil {
    return "", err
}
if !semver.IsValid(m.Version) {
    if !gover.ModIsValid(m.Path, m.Version) {
        return "", fmt.Errorf("non-semver module version %q", m.Version)
    }
if module.CanonicalVersion(m.Version) != m.Version {