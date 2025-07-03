from araras.email.utils import get_credentials
def main():
    print("Testing get_credentials from araras.email.utils")
    try:
        res = get_credentials("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
