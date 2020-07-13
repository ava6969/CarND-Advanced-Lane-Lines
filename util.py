import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def calibrate_camera(glob_images_loc):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(glob_images_loc)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    img = cv2.imread(images[1])        
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    return mtx, dist, img

# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        print(src.shape)
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)
        cv2.destroyAllWindows()

    # Return the resulting image and matrix
    return warped, M

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude
    abs_sobelxY = np.power(np.square(sobelx) + np.square(sobely), 0.5)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxY/np.max(abs_sobelxY))
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return sxbinary

def warp(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse perspective transform:
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp an image using the perspective transform, M:
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines


    return out_img, left_fit, right_fit, left_fitx, right_fitx

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    ## End visualization steps ##
    
    return result, (left_fitx, right_fitx)

def get_radius_of_curvature(x_pixels):
    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ymeters_per_pixel = 30/720 
    # meters per pixel in x dimension
    xmeters_per_pixel = 3.7/700 
    
    # Get x, y values from image
    y_image_values = np.linspace(0, 719, num=720)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image since 
    # we need curvature closest to the vehicle
    y_max = np.max(y_image_values)
    
    # Get the left and right pixels
    left_x_pixel = x_pixels[0]
    right_x_pixel = x_pixels[1]
    
    # Get the left and right coefficients
    left_x_coeff = np.polyfit(y_image_values * ymeters_per_pixel, left_x_pixel * xmeters_per_pixel, 2)
    right_x_coeff = np.polyfit(y_image_values * ymeters_per_pixel, right_x_pixel * xmeters_per_pixel, 2)
       
    # Calculate radius of curvature 
    left_curvature = ((1 + (2* left_x_coeff[0] * y_max * ymeters_per_pixel + left_x_coeff[1]) ** 2) ** 1.5) / np.absolute(2 *                                    left_x_coeff[0])
    right_curvature = ((1 + (2 * right_x_coeff[0] * y_max * ymeters_per_pixel + right_x_coeff[1]) ** 2) ** 1.5) / np.absolute(2 *                                 right_x_coeff[0])
    
    return (left_curvature, right_curvature)
    
def offset(x_values, img):
    # Calculate position of the car from the centre
    left_x_values = x_values[0]
    right_x_values = x_values[1]

    # Get the centre of the lane using the poynomial equation
    lane_diff = abs(left_x_values[len(left_x_values) - 1] - right_x_values[len(left_x_values) - 1]) / 2
    lane_centre = lane_diff + left_x_values[len(left_x_values) - 1]

    # Get the centre of the car which is the centre of the image captured by the camera
    image_centre = img.shape[1] / 2

    # offset of the car from the lane centre in pixels
    offset_pixels = abs(lane_centre - image_centre)

    # offset of the car from the lane centre in meters
    xmeter_per_pixel = 3.7/700
    offset_meters = offset_pixels * xmeter_per_pixel
    
    return offset_meters
    

def original_lane_lines(warp_img, undistorted_line_image, x_line_values, MatrInv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    image_y_values = np.linspace(0, warp_img.shape[0]-1, warp_img.shape[0] )
    
    left_x_values = x_line_values[0]
    right_x_values = x_line_values[1]

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_points = np.array([np.transpose(np.vstack([left_x_values, image_y_values]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_x_values, image_y_values])))])
    points = np.hstack((left_points, right_points))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([points]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warped_image = cv2.warpPerspective(color_warp, MatrInv, (warp_img.shape[1], warp_img.shape[0]))

    # Combine the result with the original image
    original_lane_image = cv2.addWeighted(undistorted_line_image, 1, new_warped_image, 0.3, 0)
    
    return original_lane_image

def LaneFinder(img, mtx, dist, img_name=None):
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    mag_binary = mag_thresh(undist_img, sobel_kernel=3, mag_thresh=(30, 100))
    
    imshape = img.shape
    src = np.float32([(200,imshape[0]), (600, 450), (700, 450), (1200,imshape[0])])
    dst = np.float32([(300,imshape[0]), (300, 0), (1050, 0), (1050,imshape[0])])
    src = np.expand_dims(src,1)
    dst = np.expand_dims(dst,1)
    warped, M, Minv = warp(mag_binary, src, dst)
    out_img, left_fit, right_fit, _, _ = fit_polynomial(warped)
    result, x_val = search_around_poly(warped, left_fit, right_fit)
    left_curverad, right_curverad = get_radius_of_curvature(x_val)
    offset_meters = offset(x_val, img)
    original_lane_image = original_lane_lines(warped, undist_img, x_val, Minv)
    
    title =  'left radius curvature: {}, right radius curvature: {}'.format(left_curverad, right_curverad)
    title1 = 'vehicle is {} left from center'.format(offset_meters)
    cv2.putText(original_lane_image, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(original_lane_image, title1, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2, cv2.LINE_AA)
    
    if img_name is not None:
        cv2.imwrite("output_images/"+img_name+'_magbinary.jpg',mag_binary )
        cv2.imwrite("output_images/"+img_name+'_warped.jpg', warped )
        cv2.imwrite("output_images/"+img_name+'_windowed.jpg',out_img )
        cv2.imwrite("output_images/"+img_name+'_real.jpg',result )
        cv2.imwrite("output_images/"+img_name+'_final.jpg',original_lane_image )
        
    
    return original_lane_image

