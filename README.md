# pytorch-real-world-project
- ebook: 
- source code: https://github.com/curiousily/Getting-Things-Done-with-Pytorch
- post: https://curiousily.com/posts

# Chapter 01: Getting Started with PyTorch

- 1 số lưu ý:
    - việc chuyển đổi qua lại giữa numpy và tensor ko phát sinh `cost on the performance of your app` vì NumPy and PyTorch store data in memory in the same way
    - Almost all operations have an in-place version - the name of the operation, followed by an `underscore` (`_`), vd `add_`, `unsqueeze_`,...

# Chapter 02: Build Your First Neural Network with PyTorch

- `nn.BCELoss`:  It expects the values to be outputed by the `sigmoid function`.
- `nn.BCEWithLogitsLoss` => This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.

# Chapter 03: image classification with torchvision

- vd của các kiểu transform trong torchvision: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#randomresizedcrop
- `torchvision.utils.make_grid` => dùng để ghép nhiều ảnh thành 1 ảnh lớn (như func stackimages đã sử dụng trong opencv)
- nên tham khảo bài này do hdan kỹ hơn bài trong sách: https://stackabuse.com/image-classification-with-transfer-learning-and-pytorch/
```
ảnh input đầu vào fai là RGB (thay vì BGR như khi sử dụng opencv)
```


# Chapter 04: Timeseries forecast using LSTM

# Chapter 05: Timeseries anomaly detection using lstm autoencoder
- `sequitur`: library that lets you create and train an autoencoder for sequential data (https://github.com/shobrook/sequitur)
- tương tự như sử dụng PCA để thực hiện anomaly detection => tính reconstruction loss của sample => chọn threshold để xác định anomaly
- tham khảo lstm autoencoder sử dụng keras: https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf

- tham khảo bert:
    - bert explain: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
    - bert fine-tuning: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

# Chapter 06: Detectron2
- Detectron2 là một nền tảng mã nguồn mở của Facebook AI dùng để object detection, dense pose, segmentation, ... code bằng PyTorch.

## giải thích mAP
- chú ý mAP ở trong object detection khác vs mAP@k (có nói đến trong sách `approach any ml problem`)
- mAP đơn giản là trung bình AP score của n class
- với tập dữ liệu MS COCO thì điểm mAP cuối cùng là giá trị trung bình mAP ứng với các ngưỡng `IoU` khác nhau, ví dụ mAP@[0.5:0.05:0.95] là điểm mAP trung bình ứng với ngưỡng IoU (0.5, 0.55, 0.6, ...0.95), step = 0.05
- AP là thông số trung bình precision ứng với các mốc recall tương ứng, trong đoạn từ 0 đến 1, có 2 cách tính: theo 11-points-interpolated hoặc all-points-interpolated
    - The general definition for the Average Precision (AP) is finding the area under the precision-recall curve.

- tham khảo => mình đã đọc explain về mAP ở 2 link này, khá dễ hiểu:
    - https://viblo.asia/p/deep-learning-thuat-toan-faster-rcnn-voi-bai-toan-phat-hien-duong-luoi-bo-faster-rcnn-object-detection-algorithm-for-nine-dash-line-detection-bJzKmREOZ9N
    - https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

    - ngoài ra có thể đọc thêm ở https://github.com/rafaelpadilla/Object-Detection-Metrics và new version: https://github.com/rafaelpadilla/review_object_detection_metrics

## implement detectron2
- mình chưa setup đc detectron2 trên pc cá nhân, vì vậy mình chỉ đọc hdan. theo mình thấy có 1 số bước:
    - tạo và register dataset theo format của detectron2: (Detectron chỉ nhận COCO format)
    - modify train config
    - => train

- tham khảo: 
    - instance segmentation sử dụng detectron2: https://viblo.asia/p/state-of-the-art-instance-segmentation-chi-vai-dong-code-voi-detectron2-vyDZO7W9Zwj#_register-dataset-4
    - pose estimation sử dụng lstm và keypoint detector (sử dụng pretrained detectron): https://viblo.asia/p/nhan-dien-hanh-dong-nguoi-qua-detectron2-va-lstm-1Je5E6xYKnL

# Chapter 07: Scraping Google Play App Reviews
- chỉ sử dụng lib để get data, ko fai scraping manual

# Chapter 08: sentiment analysis using bert
- The BERT authors have some recommendations for fine-tuning:
    - Batch size: 16, 32
    - Learning rate (Adam): 5e-5, 3e-5, 2e-5
    - Number of epochs: 2, 3, 

# Chapter 09: deploy using fastAPI
- video: https://www.youtube.com/watch?v=K4rRyAIn0R0&t=1s
- source: https://github.com/curiousily/Deploy-BERT-for-Sentiment-Analysis-with-FastAPI
- fastAPI ML quickstart using venv, docker & docker-compose: https://github.com/cosmic-cortex/fastAPI-ML-quickstart
- guide: https://colab.research.google.com/drive/1XITo3iuFgYfGia68NNdHwFqAfZfB3DRs?authuser=1


1 số lib mà tác giả đã sử dụng:
- `pipenv`: lib dùng để tạo venv, nhược điểm là install lib khá lâu, tuy nhiên có vẻ quản lý việc install lib khá ổn
    - thay vì dùng requirements.txt thì `pipenv` dùng Pipefile và Pipefile.lock
    - trong venv thì cần tạo env trước, sau đó cài lib, còn khi chạy `pipenv install` thì nó sẽ tạo env và install các lib đc khai báo trong Pipefile
    - `pipenv install` sẽ install lib ở virtual env => vì vậy ko ảnh hưởng đến global env, ngoài ra có thể remove env nếu lỗi
    - để sử dụng venv, có thể `pipenv shell` hoặc `pipenv run python xxx.py`
    - tham khảo:
        - Pipenv playground: https://rootnroll.com/d/pipenv/
        - https://mattgosden.medium.com/pipenv-for-easier-virtual-environments-69e1e520cde8


- `pyenv` => giúp dễ dàng switch giữa các version python. Tác giả có dùng cái này để switch qua version python khác
    - tham khảo: https://github.com/pyenv/pyenv

- `pydantic` => check data validatation
- `uvicorn` => cho phép xử lý async nhiều requests (đã đọc trong book: aproach any ml problem)

# Chapter 10: YOLO v5
- để training yolov5 với custom data, cần:
    - clone yolov5yolov5 repos: https://github.com/ultralytics/yolov5
    - chuẩn bị data, create `dataset.yaml`, create label và convert về format mà yolo yêu cầu
    - chọn model (weights) và train

=> đợi train xong và sử dụng 

- tại thời điểm mình làm, `roboflow` vừa tích hợp vs yolov5 để custom training dễ dàng hơn: https://app.roboflow.com/son-trinh, https://blog.roboflow.com/getting-started-with-roboflow/

```
so sánh yolov4 và yolov5: https://blog.roboflow.com/yolov4-versus-yolov5/
```

- tham khảo:
    - original: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
    - how to train yolov5 custom data (roboflow): https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/
    - how to train yolov4 custom data (roboflow): https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/