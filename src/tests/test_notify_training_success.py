from araras.email.utils import notify_training_success
def main():
    print("Testing notify_training_success from araras.email.utils")
    try:
        res = notify_training_success("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
