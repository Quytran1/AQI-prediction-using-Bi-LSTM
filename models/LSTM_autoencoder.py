import torch
import torch.nn as nn
import numpy as np


class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_shape):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_shape[1], hidden_size=32, bidirectional=True)
        self.decoder = nn.LSTM(input_size=64, hidden_size=32, bidirectional=True)
        self.fc = nn.Linear(64, input_shape[1])  # Output size should match input size
        
    def forward(self, x):
        x, _ = self.encoder(x)
        x = torch.repeat_interleave(x[-1:], x.size(0), dim=0)  # Repeat the last hidden state
        x, _ = self.decoder(x)
        x = self.fc(x)
        return x
    

if __name__=='__main__':
    # Giả sử chúng ta có 100 chuỗi dữ liệu, mỗi chuỗi có 10 bước thời gian, và mỗi bước thời gian có 3 đặc trưng
    data = np.random.rand(100, 10, 3)  # Dữ liệu được tạo ngẫu nhiên
    data = torch.tensor(data, dtype=torch.float32)
    # Tạo một mô hình LSTM autoencoder với kích thước đầu vào là (10, 3) (số bước thời gian và số đặc trưng)
    model = LSTM_Autoencoder(input_shape=(10, 3))
    reconstructed_data = model(data)

    # In kích thước của dữ liệu đã giải mã
    print("Kích thước của dữ liệu giải mã:", reconstructed_data.size())

# Trong ví dụ này, chúng ta tạo ngẫu nhiên một bộ dữ liệu gồm 100 chuỗi,
#  mỗi chuỗi có 10 bước thời gian và 3 đặc trưng. Sau đó, chúng ta tạo một mô hình LSTM autoencoder và sử dụng nó để mã hóa và giải mã dữ liệu. 
# Cuối cùng, chúng ta in kích thước của dữ liệu đã giải mã để đảm bảo rằng kích thước của dữ liệu đầu ra giống với dữ liệu đầu vào.