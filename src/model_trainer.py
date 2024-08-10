import tensorflow as tf
from face_model import build_model, train_model

def main():
    model = build_model(num_classes=2) 
    train_model(model, train_dir="eval", val_dir="eval")

if __name__ == "__main__":
    main()

