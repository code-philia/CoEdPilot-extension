def reverse(string, start, end):
    """
    Reverse a substring within the string from index start to end (inclusive).
    """
    while start < end:
        string[start], string[end] = string[end], string[start]
        start += 1
        end -= 1
    return string

def reverseWords(self, s: str) -> str:
    """
    This function reverses the order of words in a string.
    """