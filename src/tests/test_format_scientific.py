from araras.utils.misc import format_scientific
def main():
    print("Testing format_scientific from araras.utils.misc")
    try:
        res = format_scientific("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
