from araras.email.utils import send_email


def main():
    print("Testing send_email from araras.email.utils")
    try:
        res = send_email(
            "Hello",
            "Test body",
            "recipients.json",
            "credentials.json",
            "plain"
        )
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
