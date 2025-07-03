from araras.email.utils import get_recipient_emails
def main():
    print("Testing get_recipient_emails from araras.email.utils")
    try:
        res = get_recipient_emails("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
