func (r *toolchainRepo) Stat(ctx context.Context, rev string) (*RevInfo, error) {
	// Convert rev to DL version and stat that to make sure it exists.
	// In theory the go@ versions should be like 1.21.0
	// and the toolchain@ versions should be like go1.21.0
	// but people will type the wrong one, and so we accept
	// both and silently correct it to the standard form.
	prefix := ""
	v := rev
	v = strings.TrimPrefix(v, "go")
	if r.path == "toolchain" {
		prefix = "go"
	}

	if !gover.IsValid(v) {
		return nil, fmt.Errorf("invalid %s version %s", r.path, rev)
	}
	// If we're asking about "go" (not "toolchain"), pretend to have
	// all earlier Go versions available without network access:
	// we will provide those ourselves, at least in GOTOOLCHAIN=auto mode.
	if r.path == "go" && gover.Compare(v, gover.Local()) <= 0 {
	}
	if gover.IsLang(v) {
		return nil, fmt.Errorf("go language version %s is not a toolchain version", rev)
}