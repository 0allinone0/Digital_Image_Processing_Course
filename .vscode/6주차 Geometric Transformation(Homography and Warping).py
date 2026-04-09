import cv2
import numpy as np
import matplotlib.pyplot as plt

points = []  #유저가 클릭한 점 저장(여기서는 총 4개)

#클릭으로 기준점 지정
#사용자가 클릭을 할때 호출되는 함수
def mouse_callback(event, x, y, flags, params):
    global points  #바깥쪽에 있는 리스트를 사용하기 위해
    if event == cv2.EVENT_LBUTTONDOWN:   #왼쪽 클릭을 했을때
        if len(points) < 4:   #최대 4개의 점 클릭가능
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

img_path = 'billboard.jpg'  #이미지 path

cv2.namedWindow("Select 4 Points")   #이름이 Select 4 Points 창을 만듦
cv2.setMouseCallback("Select 4 Points", mouse_callback)   #Select 4 Points 창에서 마우스 이벤트가 발생하면 mouse_callback함수 호출

dst_img = cv2.imread(img_path)    #이미지를 읽어서 배열로 저장  (Destination Image)
print(dst_img)
print(dst_img.shape)
display_img = dst_img.copy()  #화면에 보여줄 복사본


#점 4개를 클릭할때까지의 반복문
while True:
    for point in points:
        cv2.circle(display_img, point, 5, (0, 255, 0), -1)    #각 클릭 위치마다 초록색 원을 그려주는 코드
    
    cv2.imshow("Select 4 Points", display_img)   #현재 화면을 보여줌

    if len(points) == 4:
        print("4 points have been selected.")
        break

    key = cv2.waitKey(1)     #사용자가 q를 쳤을때 멈춤
    if key == ord('q'):
        break

cv2.destroyAllWindows()   #Cv로 띄운 모든 창 닫기

print("Final selected points:", points)   


#####################
#Warping을 이용해서 이미지를 티비에 넣기

lenna = cv2.imread("Lenna.png")   #붙여넣응ㄹ 원본이미지    

h, w, _ = lenna.shape   #이미지 크기 가져오기(height, width, channel)
h_, w_, _ = dst_img.shape   #배경 이미지 크기 가져오기

src_point = np.array([[0,0], [w,0], [0,h], [w,h]])   #넣을 이미지의 꼭짓점 정의
dst_point = np.array(points).astype(np.int32)   #사용자가 클릭한 4개의 점을 배열로 바꿈 

H, status = cv2.findHomography(src_point, dst_point)    #Homography 구하기

warped_image = cv2.warpPerspective(lenna, H, (w_, h_))   #호모그래피 H를 적용해서 새로운 warped image 만들기  -> Lenna 이미지 외에 주변은 검정색 0임.
cv2.imshow("Warped image", warped_image)    #warped imaage 보여주기  

#Blending
mask = (warped_image == 0) * 1.   #점은 왜 붙이지?  -> 실수 1을 뜻함  -> Boolean값을 숫자로 바꾸는 역할
combined = (1-mask) * warped_image + dst_img * mask  #Lenna가 들어간 부분만 바꾸고 나머지는 유지하는 것

cv2.imshow("Final image", combined/255.)
cv2.waitKey(0)
cv2.destroyAllWindows()

