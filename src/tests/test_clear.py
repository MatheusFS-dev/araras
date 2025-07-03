from araras.utils.misc import clear
def main():
    print("Testing clear from araras.utils.misc")
    try:
        res = clear()
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
