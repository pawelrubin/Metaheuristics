import local_search as ls

if __name__ == "__main__":
    print(ls.iterated_local_search(4, ls.happy_cat, 3)[-1])
    print(ls.iterated_local_search(4, ls.griewank, 3)[-1])
