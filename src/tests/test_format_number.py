from araras.utils.misc import format_number
def main():
    print("Testing format_number from araras.utils.misc")
    try:
        res = format_number("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
