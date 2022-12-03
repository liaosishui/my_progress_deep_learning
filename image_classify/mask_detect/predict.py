import torchvision
import matplotlib.pyplot as plt

def pred(net, image, image_type = 'noTensor', show_img = True):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((240, 240)),
        torchvision.transforms.ToTensor()
    ])
    if image_type == 'noTensor':
        image_tensor = transform(image)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    labels = ['nomaskimage', 'maskimage']
    y_hat = net(image_tensor).argmax()
    y_pred = labels[y_hat]
    print(y_pred)
    if show_img == True:
        image_tensor = image_tensor.squeeze(0)
        plt.imshow(image_tensor.permute((1, 2, 0)))
        plt.title(y_pred)
        print(image_tensor.shape)
    return y_hat
