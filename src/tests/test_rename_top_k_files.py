from araras.optuna.utils import rename_top_k_files
def main():
    print("Testing rename_top_k_files from araras.optuna.utils")
    try:
        res = rename_top_k_files("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
