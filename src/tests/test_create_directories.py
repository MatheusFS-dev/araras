from araras.optuna.analyze import create_directories
def main():
    print("Testing create_directories from araras.optuna.analyze")
    try:
        res = create_directories("dummy", "dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
