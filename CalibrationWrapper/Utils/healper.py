from CameraCalibration import *
import json

def load_yml(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return False

    if fs.getNode("M1").empty() or fs.getNode("M2").empty():
        return False

    M1 = fs.getNode("M1").mat()
    d1 = fs.getNode("d1").mat()
    M2 = fs.getNode("M2").mat()
    d2 = fs.getNode("d2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    E = fs.getNode("E").mat()
    F = fs.getNode("F").mat()
    
    calibrationError = fs.getNode("calibrationError").real()
    
    imageSize = fs.getNode("imageSize").mat().flatten().tolist()
    imageSize = (int(imageSize[0]), int(imageSize[1])) # should be 1024 * 768
    
    t = fs.getNode("t").mat().flatten().tolist()
    t = tuple(t)
    
    angle = fs.getNode("angle").real()
    
    gridsize = fs.getNode("gridsize").mat().flatten().tolist()
    gridsize = (int(gridsize[0]), int(gridsize[1]))
    
    fs.release()

    # The first list contains things that we need.
    # The second list contains things that we don't need yet.
    return [M1, d1, M2, d2, R, T, imageSize], [t, angle]

def saveInJSON(filePath, l1, l2):
    """
    <type 'numpy.ndarray'> <type 'numpy.ndarray'> <type 'numpy.ndarray'> <type 'numpy.ndarray'> <type 'numpy.ndarray'> <type 'numpy.ndarray'> <type 'tuple'>
    <type 'tuple'> <type 'float'>
    """
    
    # Converting all the params from numpy arrays to python lists
    cam1_intrinsics_json = l1[0].tolist()
    cam2_intrinsics_json = l1[2].tolist()
    cam1_distortion_json = l1[1].tolist()
    cam2_distortion_json = l1[3].tolist()
    cam2_rotation_json = l1[4].tolist()
    cam2_translation_json = l1[5].tolist()
    imgSize = [l1[6][1], l1[6][0]]

    t_json = [l2[0][0], l2[0][1]]
    angle_json = [l2[1]]

    # Camera 1 params dictionary
    cam1_params = {}
    cam1_params['intrinsic'] = cam1_intrinsics_json
    cam1_params['distortion'] = cam1_distortion_json

    # Camera 2 params dictionary
    cam2_params = {}
    cam2_params['intrinsic'] = cam2_intrinsics_json
    cam2_params['distortion'] = cam2_distortion_json

    # Camera 2 rotation and translation
    cam2_rot_tran = {}
    cam2_rot_tran['rotation'] = cam2_rotation_json
    cam2_rot_tran['translation'] = cam2_translation_json

    # Image Size
    image_params = {}
    image_params['size'] = imgSize

    # translation and rotation
    calibration_params = {}
    calibration_params['t'] = t_json
    calibration_params['angle'] = angle_json
    
    #Final JSON data
    data = {}
    data['camera1'] = cam1_params
    data['camera2'] = cam2_params
    data['camera2RotateTranslate'] = cam2_rot_tran
    data['globalTAngle'] = calibration_params
    data['imgSize'] = image_params


    if len(cam1_params) == 0:
        print("Camera 1 parameters are not saved correctly")
    if len(cam2_params) == 0:
        print("Camera 2 parameters are not saved correctly")


    with open(filePath, 'w') as f:
        try:
            json.dump(data, f)
            print("JSON file saved")
        except Exception as ex:
            print("got %s on saving json file" % ex)

def parseJSON(filePath):
    # Loading json file
    with open(filePath) as json_file:
        try:
            data = json.load(json_file)
            print('JSON file is loaded: ', json_file)
        except Exception as ex:
            print("got %s on json.load" % ex)

    # TODO : Add sanity check for file reading and data loading here.

    # Getting Camera1 parameters
    camera1_params = data['camera1']
    cam1_intrinsic_param = camera1_params['intrinsic']
    cam1_distortion = camera1_params['distortion']

    # Getting Camera2 parameters
    camera2_params = data['camera2']
    cam2_intrinsic_param = camera2_params['intrinsic']
    cam2_distortion = camera2_params['distortion']

    # Getting Camera 2 rotation and translation
    camera2RotTran = data['camera2RotateTranslate']
    cam2_rotation = camera2RotTran["rotation"]
    cam2_translation = camera2RotTran["translation"]

    # Getting image size
    size = data['imgSize']['size']

    # Converting json list into correct data format
    cam1_intrinsic_param = np.array(cam1_intrinsic_param)
    cam2_intrinsic_param = np.array(cam2_intrinsic_param)
    cam1_distortion = np.array(cam1_distortion)
    cam2_distortion = np.array(cam2_distortion)
    cam2_rotation = np.array(cam2_rotation)
    cam2_translation = np.array(cam2_translation)
    imageSize = (size[1], size[0])

    print(cam1_intrinsic_param)
    print(cam2_intrinsic_param)
    print(cam1_distortion)
    print(cam2_distortion)
    print(cam2_rotation)
    print(cam2_translation)
    print(imageSize)

    # return cam1_intrinsic_param, cam2_intrinsic_param, cam1_distortion, cam2_distortion, cam2_rotation, cam2_translation, imageSize


if __name__ == "__main__":
    
    base_dir = ""
    
    # Create matlab.yml file
    # test(base_dir)

    # Convert .yml file into .json file and save it
    l1, l2 = load_yml("matlab.yml")
    saveInJSON('camera_parameters_binning_v2.json', l1, l2)

    # Cheack the json file is correct
    parseJSON('camera_parameters_binning_v2.json')


