from araras.utils.misc import format_bytes
def main():
    print("Testing format_bytes from araras.utils.misc")
    try:
        res = format_bytes("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
