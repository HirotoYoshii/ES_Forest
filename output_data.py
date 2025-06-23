import struct
import mnist as dl

if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = dl.load_mnist()
    print(len(X_test))
    print(max(X_train[0]))
    print(min(X_train[0]))

    sample_num, dimension_num = X_train.shape

    with open("data_train.bin", "wb") as f:
      f.write(struct.pack("i", sample_num))
      f.write(struct.pack("i", dimension_num))    

      for arr in X_train:
        for val in arr:
          f.write(struct.pack("f", val))

    with open("label_train.bin", "wb") as f:
      f.write(struct.pack("i", sample_num))
      for val in y_train:
        f.write(struct.pack("b", val))

    sample_num, dimension_num = X_test.shape
    with open("data_test.bin", "wb") as f:
      f.write(struct.pack("i", sample_num))
      f.write(struct.pack("i", dimension_num))
      
      for arr in X_test:
        for val in arr:
          f.write(struct.pack("f", val))

    with open("label_test.bin", "wb") as f:
      f.write(struct.pack("i", sample_num))
      for val in y_test:
        f.write(struct.pack("b", val))
