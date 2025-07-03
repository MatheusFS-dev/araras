from araras.email.utils import notify_warning
def main():
    print("Testing notify_warning from araras.email.utils")
    try:
        res = notify_warning("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
