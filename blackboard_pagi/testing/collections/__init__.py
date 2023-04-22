def unordered_equal(iter_a, iter_b):
    list_b = list(iter_b)

    for a in iter_a:
        if a not in list_b:
            return False
        list_b.remove(a)

    if list_b:
        return False
    return True
