from araras.optuna.analyze import format_title
def main():
    print("Testing format_title from araras.optuna.analyze")
    try:
        res = format_title("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
