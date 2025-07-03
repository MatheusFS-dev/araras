from araras.plot.configs import config_plt
def main():
    print("Testing config_plt from araras.plot.configs")
    try:
        res = config_plt("dummy")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
