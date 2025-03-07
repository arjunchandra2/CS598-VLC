import cv2
from skimage.metrics import structural_similarity as ssim

# Load images in grayscale
image1 = cv2.imread("/projectnb/cs598/students/ac25/CS598-VLC/style_transfer_experiments/adain_test/000000.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("/projectnb/cs598/students/ac25/CS598-VLC/style_transfer_experiments/hmdb51_sample_2/sample.jpg", cv2.IMREAD_GRAYSCALE)

# Resize image2 to match image1
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Compute SSIM
score, _ = ssim(image1, image2, full=True)
print(f"SSIM Score: {score}")
