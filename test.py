


def main():
    # create text dataset
    sentences, labels = create_dataset()

    model = SentenceTransformer("vinai/phobert-base-v2")
    model.max_seq_length = 128

    
    



if __name__ == "__main__":
    main()
