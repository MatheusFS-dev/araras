from araras.utils.gpu import _get_nvidia_smi_data
def main():
    print("Testing _get_nvidia_smi_data from araras.utils.gpu")
    try:
        res = _get_nvidia_smi_data()
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
