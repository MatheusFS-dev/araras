from araras.email.utils import run_with_notification


def dummy_task(x, y):
    print("Running dummy task")
    return x + y


def main():
    print("Testing run_with_notification from araras.email.utils")
    try:
        res = run_with_notification(
            dummy_task,
            (1, 2),
            {},
            "recipients.json",
            "credentials.json",
            "Done",
            "<p>success</p>",
            "plain"
        )
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
