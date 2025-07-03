from araras.utils.misc import format_number_commas
def main():
    print("Testing format_number_commas from araras.utils.misc")
    try:
        res = format_number_commas("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
