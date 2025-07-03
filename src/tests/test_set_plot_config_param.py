from araras.optuna.analyze import set_plot_config_param
def main():
    print("Testing set_plot_config_param from araras.optuna.analyze")
    try:
        res = set_plot_config_param("dummy", "dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
