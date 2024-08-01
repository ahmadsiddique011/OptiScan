import customtkinter # GUI library
import os
from PIL import Image # library for accessing images
from HelperFunctions import get_screen_size # function from file to get width and height of screen
from tkinter import filedialog, messagebox # GUI library
import report # file for scan and generate report

# variable for tracking which image map is active in GUI
activeBtn = ""


# function for changing selection of image map in GUI
def changeActive(name):
    global activeBtn # accessing activeBtn from outside of function
    activeBtn = name # changing activebtn value to current selected image map in GUI

    changeImage() # changing image in GUI
    UIUpdateOnChangeActive() # changing the outside from previous selected to current selected

# function for clearing outisde from buttons except the selected button
def UIUpdateOnChangeActive():
    leftFourMap.configure(border_width= 3 if activeBtn == "leftFourMap" else 0) # left side 4 map button
    leftEnhancedEstasia.configure(border_width= 3 if activeBtn == "leftEnhancedEstasia" else 0) # left side enhanced estasia button
    leftFourierAnalysis.configure(border_width= 3 if activeBtn == "leftFourierAnalysis" else 0) # left side fourier analysis button
    leftZernikCorneaBack.configure(border_width= 3 if activeBtn == "leftZernikCorneaBack" else 0) # left side zernik back button
    leftZernikCorneaFront.configure(border_width= 3 if activeBtn == "leftZernikCorneaFront" else 0) # left side zernik front button
    rightFourMap.configure(border_width= 3 if activeBtn == "rightFourMap" else 0) # right side 4 map button
    rightEnhancedEstasia.configure(border_width= 3 if activeBtn == "rightEnhancedEstasia" else 0) # right side enhanced estasia button
    rightFourierAnalysis.configure(border_width= 3 if activeBtn == "rightFourierAnalysis" else 0) # right side fourier analysis button
    rightZernikCorneaBack.configure(border_width= 3 if activeBtn == "rightZernikCorneaBack" else 0) # right side zernik back button
    rightZernikCorneaFront.configure(border_width= 3 if activeBtn == "rightZernikCorneaFront" else 0) # right side zernik front button
    

# dic to save the images path to use in AI and to show images in
imagesPaths = {
    "leftFourMap": "",
    "leftEnhancedEstasia": "",
    "leftFourierAnalysis": "",
    "leftZernikCorneaBack": "",
    "leftZernikCorneaFront": "",
    "rightFourMap": "",
    "rightEnhancedEstasia": "",
    "rightFourierAnalysis": "",
    "rightZernikCorneaBack": "",
    "rightZernikCorneaFront": ""
}

# function for showing green color on button when the image of coresponding button is selected
def UIUpdateOnImageSelect():
    # checking path for left side 4 map image and if image path is not "" then seting button color green
    if (imagesPaths["leftFourMap"] != ""):
        leftFourMap.configure(fg_color="green")

    # checking path for left side enhanced estasia image and if image path is not "" then seting button color green
    if (imagesPaths["leftEnhancedEstasia"] != ""):
        leftEnhancedEstasia.configure(fg_color="green")
    
    # checking path for left side fourier analysis image and if image path is not "" then seting button color green
    if (imagesPaths["leftFourierAnalysis"] != ""):
        leftFourierAnalysis.configure(fg_color="green")
    
    # checking path for left side fzernik back image and if image path is not "" then seting button color green
    if (imagesPaths["leftZernikCorneaBack"] != ""):
        leftZernikCorneaBack.configure(fg_color="green")

    # checking path for left side fzernik front image and if image path is not "" then seting button color green
    if (imagesPaths["leftZernikCorneaFront"] != ""):
        leftZernikCorneaFront.configure(fg_color="green")
    
    # checking path for right side 4 map image and if image path is not "" then seting button color green
    if (imagesPaths["rightFourMap"] != ""):
        rightFourMap.configure(fg_color="green")
    
    # checking path for right side enhanced estasia image and if image path is not "" then seting button color green
    if (imagesPaths["rightEnhancedEstasia"] != ""):
        rightEnhancedEstasia.configure(fg_color="green")
    
    # checking path for right side fourier analysis image and if image path is not "" then seting button color green
    if (imagesPaths["rightFourierAnalysis"] != ""):
        rightFourierAnalysis.configure(fg_color="green")
    
    # checking path for right side fzernik back image and if image path is not "" then seting button color green
    if (imagesPaths["rightZernikCorneaBack"] != ""):
        rightZernikCorneaBack.configure(fg_color="green")
    
    # checking path for right side fzernik front image and if image path is not "" then seting button color green
    if (imagesPaths["rightZernikCorneaFront"] != ""):
        rightZernikCorneaFront.configure(fg_color="green")

    changeImage() # changing image on GUI


# function for changing image on GUI
def changeImage():
    global not_selected_image_label # accessing variable from outisde function
    
    not_selected_image_label.destroy() # deleting image from GUI

    # checking if image path does not exist of selected image button
    if (imagesPaths[activeBtn] == ""):
        # then getting default image for light theme
        lightImage = Image.open(os.path.join(os.getcwd(), 'assets', 'no_image_light.png'))
        # then getting default image for dark theme
        darkImage = Image.open(os.path.join(os.getcwd(), 'assets', 'no_image_dark.png'))
        # adding image into label
        not_selected_image = customtkinter.CTkImage(light_image=lightImage, dark_image=darkImage, size=(width / 2.5, height / 2.45))
        # activating label
        not_selected_image_label = customtkinter.CTkLabel(leftFrame, image=not_selected_image, text="", corner_radius=100)
        # setting label position.
        not_selected_image_label.grid(row=0, column=0, padx=10, pady=(0, 20))
    else:
        # else getting image from images path
        # same image will be used for light and dark theme.
        lightImage = Image.open(imagesPaths[activeBtn])
        darkImage = Image.open(imagesPaths[activeBtn])
        # adding image into label
        not_selected_image = customtkinter.CTkImage(light_image=lightImage, dark_image=darkImage, size=(width / 2.5, height / 2.45))
        # activating label
        not_selected_image_label = customtkinter.CTkLabel(leftFrame, image=not_selected_image, text="", corner_radius=100)
        # setting label position.
        not_selected_image_label.grid(row=0, column=0, padx=10, pady=(0, 20))

    
# function for opening image select box on clicking "Browse Image" button.
def browseImage():
    # checking if activeBtn is "", and if it is True then showing error message.
    # checking this because the specific image button have to be activated for correctly saving image in imagesPath
    if (activeBtn == ""):
        return messagebox.showerror("Error", "Please select an image type from side panel.")
    
    # try and except for error handling, otherwise the application will close from error
    try:
        # opening file select box to select jpg type or all type files
        file = filedialog.askopenfilename(title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        # splitting image name into parts on basis ".", then selecting last part
        fileExt = file.split(".")[-1]

        # checking if last part is not jpg
        if (fileExt.lower() != 'jpg'):
            # then telling user to only select jpg type file
            return messagebox.showerror("Incorrect file", "Please select a .jpg file")
        
        # if everything goes correct, then saving selected image path
        imagesPaths[activeBtn] = file

    except:
        # showing user error, if an unknown error occurec
        return messagebox.showerror("Error", "Unknown error occured. Please try again.")
    
    # calling this function to change button color green and show selected image into GUI
    UIUpdateOnImageSelect()

# function to start scan and generate report
def generateReport():
    try:
        pdf, first_name, last_name = report.report(imagesPath=imagesPaths)

        # after scanning, asking user to select folder to save pdf
        messagebox.showinfo("Select Folder", "Please select folder for saving report pdf.")

        # opening select folder box and saving selected folder path
        dir = filedialog.askdirectory()

        # path with pdf name, name is  (first-name)_(last-name)_Report.pdf
        pdfPath = os.path.join(dir, f'{first_name}_{last_name}_Report.pdf')

        # saving pdf in folder
        pdf.output(pdfPath)

        # telling user that pdf is saved
        messagebox.showinfo("PDF saved", "Report generated successfully.")
        
        # open the folder in which the pdf is saved.
        os.startfile(dir)
    except:
        # if error, then telling user error
        messagebox.showerror("Error", "Error occured try again.")

    backFrame.destroy()
    scanFrame.destroy()

# function to check images and then show scanning message and in last start scanning after 500ms or half second
def startScan():
    # checking if all images paths are present
    # if not then show message to the user to select all images
    for path in imagesPaths.values():
        if (path == ""):
            return messagebox.showerror("Error", "Please select all images.")
    
    backFrame.place(x=0, y=110)
    # Frame using place with x, y
    x = (width / 3) - 200
    y = (height / 3) - 80
    scanFrame.place(x=x, y=y)
    app.after(500, generateReport)

# activating GUI library and starting GUI window
app = customtkinter.CTk()

# setting application name on top of window
app.title("Optiscan")

# getting theme mode i.e. dark, light
themeMode = app._get_appearance_mode()

# getting width and height
width, height = get_screen_size() # get screen size

# setting size of application to half width of screen width and half height of screen height
app.geometry(f'{int(width / 2)}x{int(height / 2)}') # set window size

# disalabing resizing application 
app.resizable(False, False) # disable resizing
# making application full screen
app.after(0, lambda:app.state('zoomed')) # maximize the window
# setting icon on application
app.iconbitmap(os.path.join(os.getcwd(), 'assets', 'optiscanIcon.ico'))

# ===============================================================================================
# Header frame for logo and application name
# ===============================================================================================
header = customtkinter.CTkFrame(app, fg_color=("#ebebeb", "#242424"))
header.pack(padx=0, pady=0)

# ===============================================================================================
# Image to show application Logo
# ===============================================================================================
# image for logo
logo = Image.open(os.path.join(os.getcwd(), 'assets', 'optiscan logo.png'))
# adding images and setting size
logo_image = customtkinter.CTkImage(light_image=logo, dark_image=logo, size=(200/2, 150/2))
# activating label with images
logo_image_label = customtkinter.CTkLabel(header, image=logo_image, text="", corner_radius=100)
# setting label position
logo_image_label.grid(row=0, column=0)

# ===============================================================================================
# label to show application name
# ===============================================================================================
label = customtkinter.CTkLabel(header, text="OPTISCAN", font=("", 45), padx=20, pady=50)
label.grid(row=0, column=1, pady=(0, 10))


# ===============================================================================================
# Main frame
# ===============================================================================================
frame = customtkinter.CTkFrame(app, fg_color=("silver", "#2a2a2a"))
frame.pack(expand=True, fill="both", padx=0, pady=0)
frame.grid_rowconfigure(0, weight=1)


# ===============================================================================================
# Frames to show scanning message
# ===============================================================================================
backFrame = customtkinter.CTkFrame(app, width=width, height=height/1.3, fg_color="transparent")
scanFrame = customtkinter.CTkFrame(app, width=600, height=160, fg_color=("#2a2a2a", "white"))
# label to show text "Please wait, Scanning..."
label = customtkinter.CTkLabel(scanFrame, text="Please wait, Scanning...", font=("", 26), width=590, height=150, fg_color=("#ebebeb", "#242424"))
label.pack(padx=5, pady=5)


# ===============================================================================================
# frame for showing image on left side.
# ===============================================================================================
leftFrame = customtkinter.CTkFrame(frame, fg_color=("silver", "#2a2a2a"))
leftFrame.grid(row=0, column=0, padx=(int(width * 0.1), 0), pady=(50, 0), sticky='n')
leftFrame.columnconfigure(0, weight=1)

# importing images for showing that image is not selected
# light image for light theme
lightImage = Image.open(os.path.join(os.getcwd(), 'assets', 'no_image_light.png'))
# dark image for dark theme
darkImage = Image.open(os.path.join(os.getcwd(), 'assets', 'no_image_dark.png'))
# adding images and setting size
not_selected_image = customtkinter.CTkImage(light_image=lightImage, dark_image=darkImage, size=(width / 2.5, height / 2.45))
# activating label with images
not_selected_image_label = customtkinter.CTkLabel(leftFrame, image=not_selected_image, text="", corner_radius=100)
# setting label position
not_selected_image_label.grid(row=0, column=0, padx=10, pady=(0, 20))

# button for Browse Image with function browseImage
image_browse_btn = customtkinter.CTkButton(leftFrame, width=120, height=int(height * 0.045), text="Browse Image", command=browseImage)
# setting button position
image_browse_btn.grid(row=1, column=0, padx=10, pady=(10, 0))
    
# ===============================================================================================
# frame for showing tabs on right side.
# ===============================================================================================
rightFrame = customtkinter.CTkFrame(frame, fg_color=("silver", "#2a2a2a"))
rightFrame.grid(row=0, column=1, pady=(38, 0), sticky='n')
rightFrame.columnconfigure(0, weight=1)


# ===============================================================================================
# Functions for toggle left eye and right eye.
# ===============================================================================================
# function for activating right side eye buttons
def showRightEyeFrame():
    leftEyeFrame.grid_forget()
    rightEyeFrame.grid(row=0, column=1, padx=20)
    rightEyeFrame.columnconfigure(0, weight=1)

# function for activating left side eye buttons
def showLeftEyeFrame():
    rightEyeFrame.grid_forget()
    leftEyeFrame.grid(row=0, column=1, padx=20)
    leftEyeFrame.columnconfigure(0, weight=1)

# ===============================================================================================
# frame for showing images button for left eye. 
# this frame contains the button of left side eye buttons
# ===============================================================================================

# frame which hold all buttons
leftEyeFrame = customtkinter.CTkFrame(rightFrame, fg_color=("silver", "#2a2a2a"))
leftEyeFrame.grid(row=0, column=1, padx=20)
leftEyeFrame.columnconfigure(0, weight=1)

# left button for changing left side
left_LeftButton = customtkinter.CTkButton(leftEyeFrame, text="Left", height=int(height * 0.05), corner_radius=0, font=("", 20))
left_LeftButton.grid(row=0, column=0, sticky='ew', pady=10)

# right button for changing right side
left_RightButton = customtkinter.CTkButton(leftEyeFrame, text="Right", height=int(height * 0.05), corner_radius=0, fg_color="#dddddd", text_color='black', hover_color="white", font=("", 20), command=showRightEyeFrame)
left_RightButton.grid(row=0, column=1, sticky='ew', pady=10)

# left side 4 Maps Refraction button
leftFourMap = customtkinter.CTkButton(leftEyeFrame, text="4 Maps Refraction", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("leftFourMap"))
leftFourMap.grid(row=1, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# left side Enhanced Estasia button
leftEnhancedEstasia = customtkinter.CTkButton(leftEyeFrame, text="Enhanced Estasia", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("leftEnhancedEstasia"))
leftEnhancedEstasia.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# left side Fourier Analysis button
leftFourierAnalysis = customtkinter.CTkButton(leftEyeFrame, text="Fourier Analysis", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("leftFourierAnalysis"))
leftFourierAnalysis.grid(row=3, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# left side Zernik Cornea Back button
leftZernikCorneaBack = customtkinter.CTkButton(leftEyeFrame, text="Zernik Cornea Back", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("leftZernikCorneaBack"))
leftZernikCorneaBack.grid(row=4, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# left side Zernik Cornea Front button
leftZernikCorneaFront = customtkinter.CTkButton(leftEyeFrame, text="Zernik Cornea Front", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("leftZernikCorneaFront"))
leftZernikCorneaFront.grid(row=5, column=0, columnspan=2, sticky='ew', pady=10, padx=20)


# ===============================================================================================
# frame for showing images button for right eye.
# this frame contains the button of right side eye buttons
# ===============================================================================================

# frame which hold all buttons
rightEyeFrame = customtkinter.CTkFrame(rightFrame, fg_color=("silver", "#2a2a2a"))

# ======= line for testing ====================
# rightEyeFrame.grid(row=0, column=1, padx=20)
# rightEyeFrame.columnconfigure(0, weight=1)

# left button for changing left side
right_LeftButton = customtkinter.CTkButton(rightEyeFrame, text="Left", height=int(height * 0.05), corner_radius=0, fg_color="#dddddd", text_color='black', hover_color="white", font=("", 20), command=showLeftEyeFrame)
right_LeftButton.grid(row=0, column=0, sticky='ew', pady=10)

# right button for changing right side
right_RightButton = customtkinter.CTkButton(rightEyeFrame, text="Right", height=int(height * 0.05), corner_radius=0, font=("", 20))
right_RightButton.grid(row=0, column=1, sticky='ew', pady=10)

# right side 4 Maps Refraction button
rightFourMap = customtkinter.CTkButton(rightEyeFrame, text="4 Maps Refraction", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("rightFourMap"))
rightFourMap.grid(row=1, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# right side Enhanced Estasia button
rightEnhancedEstasia = customtkinter.CTkButton(rightEyeFrame, text="Enhanced Estasia", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("rightEnhancedEstasia"))
rightEnhancedEstasia.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# right side Fourier Analysis button
rightFourierAnalysis = customtkinter.CTkButton(rightEyeFrame, text="Fourier Analysis", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("rightFourierAnalysis"))
rightFourierAnalysis.grid(row=3, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# right side Zernik Cornea Back button
rightZernikCorneaBack = customtkinter.CTkButton(rightEyeFrame, text="Zernik Cornea Back", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("rightZernikCorneaBack"))
rightZernikCorneaBack.grid(row=4, column=0, columnspan=2, sticky='ew', pady=10, padx=20)

# right side Zernik Cornea Front button
rightZernikCorneaFront = customtkinter.CTkButton(rightEyeFrame, text="Zernik Cornea Front", height=40, border_color=("#2a2a2a","#dddddd"), command=lambda:changeActive("rightZernikCorneaFront"))
rightZernikCorneaFront.grid(row=5, column=0, columnspan=2, sticky='ew', pady=10, padx=20)



# ===============================================================================================
# button for scan and generate PDF
# ===============================================================================================
scanAndGenerateBtn = customtkinter.CTkButton(frame, text="Scan and Generate", height=int(height * 0.045), command=startScan)
scanAndGenerateBtn.grid(row=1, column=1, sticky='ew', pady=(0, 60), padx=20)


# this line is responsible for running GUI
app.mainloop()