from araras.optuna.analyze import get_param_display_name
def main():
    print("Testing get_param_display_name from araras.optuna.analyze")
    try:
        res = get_param_display_name("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
