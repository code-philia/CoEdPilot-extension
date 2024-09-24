export function limitNum(x: number, lower: number, upper: number) {
    if (x <= lower)
        x = lower;
    if (x >= upper)
        x = upper;
    return x;
}
