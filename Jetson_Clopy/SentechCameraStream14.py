# import the necessary packages
import stapipy as st
import threading
import cv2
import time
import numpy as np

# Feature names
PIXEL_FORMAT = "PixelFormat"
REGION_SELECTOR = "RegionSelector"
REGION_MODE = "RegionMode"
OFFSET_X = "OffsetX"
OFFSET_Y = "OffsetY"
WIDTH = "Width"
HEIGHT = "Height"


# Create global stapipy system object to allow multiple camera usage
# Initialize StApi before using.
st.initialize()
# Create a system object for device scan and connection.
st_system = st.create_system()

class CMyCallback:
    """
    Class that contains a callback function.
    """

    def __init__(self):
        self._image = None
        self._lock = threading.Lock()

    @property
    def image(self):
        duplicate = None
        self._lock.acquire()
        if self._image is not None:
            duplicate = self._image.copy()
        self._lock.release()
        return duplicate

    def datastream_callback(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that trigger the callback.
        :param context: user data passed on during callback registration.
        """
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Check the pixelformat of the input image.
                    pixel_format = st_image.pixel_format
                    pixel_format_info = st.get_pixel_format_info(pixel_format)

                    # Only mono or bayer is processed.
                    if not(pixel_format_info.is_mono or
                           pixel_format_info.is_bayer):
                        return

                    # Get raw image data.
                    data = st_image.get_image_data()

                    # Decode raw image data.
                    if pixel_format_info.each_component_total_bit_count > 8:
                        # Keep full 10/12-bit precision as uint16.
                        nparr = np.frombuffer(data, np.uint16)
                    else:
                        # Mono8
                        nparr = np.frombuffer(data, np.uint8)

                    # Mono camera: reshape to 2D, single channel.
                    nparr = nparr.reshape(st_image.height, st_image.width)

                    # Store latest frame (2D, mono, uint8 or uint16).
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()

class SentechCameraStream14:
    def __init__(self, cfgDict=None, deviceIdx=0):
        self.cfg_dict = cfgDict
        self.res = list(map(int, cfgDict['resolution'].split(', ')))
        # initialize the camera and stream
        self.my_callback = CMyCallback()
        self.cb_func = self.my_callback.datastream_callback

        # Get the interface class so that we can create device by index to support multiple cameras
        st_interface = st_system.get_interface(0)
        self.st_device = st_interface.create_device_by_index(deviceIdx)

        # Connect to first detected device.
        #self.st_device = st_system.create_first_device()

        # Display DisplayName of the device.
        print('Device=', self.st_device.info.display_name)

        # Create a datastream object for handling image stream data.
        self.st_datastream = self.st_device.create_datastream()

        # Get INodeMap object to access the setting of the device.
        remote_nodemap = self.st_device.remote_port.nodemap

        # ----------------------------------------------------
        # Configure this camera according to index and your rules
        # (PixelFormat, ExposureAuto, GainAuto, Gain, BlackLevel, Gamma)
        # ----------------------------------------------------
        self.configure_camera(remote_nodemap, deviceIdx)

        # Check and set CameraSideROI if you want
        # self.set_camera_side_roi(remote_nodemap)

        # Register callback for datastream
        callback = self.st_datastream.register_callback(self.cb_func)


        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.stopped = False
        self.updateSleep = 0.0

    def start(self):

        # Start the image acquisition of the host (local machine) side.
        self.st_datastream.start_acquisition()

        # Start the image acquisition of the camera side.
        self.st_device.acquisition_start()

        # Get device nodemap to access the device settings.
        # remote_nodemap = self.st_device.remote_port.nodemap

        # # Create and start a thread for auto function configuration.
        # autofunc_thread = threading.Thread(target=self.do_auto_functions,
        #                                    args=(remote_nodemap,))
        # autofunc_thread.start()

        return self

    def read(self):
        # the frame most recently read
        return self.my_callback.image

    def clearimg(self):
        print('clearimg not implemented')
        return

    def stop(self):

        # self.autofunc_thread.join()

        # Stop the image acquisition of the camera side
        self.st_device.acquisition_stop()

        # Stop the image acquisition of the host side
        self.st_datastream.stop_acquisition()

        # indicate that the thread should be stopped
        self.stopped = True

    def disp_pose(self, image, pose):
        for point in pose:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (255,0,0))
    

    # ------------------------------------------------------------------
    # Small helpers to set enum and numeric nodes programmatically
    # ------------------------------------------------------------------
    def _set_enum(self, nodemap, node_name, entry_symbolic):
        """
        Set an enumeration node to the given symbolic entry name.
        e.g. _set_enum(nodemap, "PixelFormat", "Mono12")
        """
        try:
            enum_node = st.PyIEnumeration(nodemap.get_node(node_name))
            entry_node = st.PyIEnumEntry(enum_node[entry_symbolic])
            enum_node.set_entry_value(entry_node)
            print(f"{node_name} = {entry_symbolic}")
        except Exception as e:
            print(f"[SentechCameraStream] Could not set {node_name} "
                  f"to {entry_symbolic}: {e}")

    def _set_numeric(self, nodemap, node_name, value):
        """
        Set a numeric node (integer or float) to value.
        e.g. _set_numeric(nodemap, "Gamma", 1.0)
        """
        try:
            node = nodemap.get_node(node_name)
            if node.principal_interface_type == st.EGCInterfaceType.IFloat:
                node_value = st.PyIFloat(node)
            else:
                node_value = st.PyIInteger(node)
            node_value.value = value
            print(f"{node_name} = {value}")
        except Exception as e:
            print(f"[SentechCameraStream] Could not set {node_name}: {e}")

    def configure_camera(self, nodemap, deviceIdx):
        """
        Configure pixel format, gain, exposure auto, black level, gamma
        for this camera.

        deviceIdx == 0  -> brain camera (Mono12, Gain 0)
        deviceIdx == 1  -> behavior camera (Mono12, Gain 10)
        """
        # ---------------- Pixel Format ----------------
        try:
            if deviceIdx == 0:
                # Brain camera
                self._set_enum(nodemap, "PixelFormat", "Mono12")
            else:
                # Behavior camera
                self._set_enum(nodemap, "PixelFormat", "Mono12")
        except Exception as e:
            print("[SentechCameraStream] PixelFormat config failed:", e)

        # ---------------- SET TARGET FPS ----------------
        if deviceIdx == 0:
            # Brain camera
            target_fps = 30
        else:
            # Behavior camera
            target_fps = 30

        # Enable and set AcquisitionFrameRate if available
        try:
            node_en = nodemap.get_node("AcquisitionFrameRateEnable")
            if node_en and node_en.is_writable:
                st.PyIBoolean(node_en).value = True
                print("AcquisitionFrameRateEnable = True")
        except Exception as e:
            print("AcquisitionFrameRateEnable not available:", e)

        try:
            self._set_numeric(nodemap, "AcquisitionFrameRate", target_fps)
        except Exception as e:
            print("AcquisitionFrameRate not set:", e)

        # ---------------- Exposure Auto OFF ----------------
        try:
            self._set_enum(nodemap, "ExposureAuto", "Off")
        except Exception as e:
            print("[SentechCameraStream] ExposureAuto config failed:", e)

        # ------------- SET EXPOSURE TIME ------------------------
        # Give each camera a sensible exposure that fits within the frame period.
        if deviceIdx == 0:
            # Brain @ 30 fps -> ~30 ms exposure
            exposure_us = 30000.0
        else:
            # Behavior @ 60 fps -> ~12–15 ms
            exposure_us = 3000.0

        # Try ExposureTime first, fall back to ExposureTimeRaw
        try:
            self._set_numeric(nodemap, "ExposureTime", exposure_us)
        except Exception as e:
            print("ExposureTime not set, trying ExposureTimeRaw:", e)
            try:
                self._set_numeric(nodemap, "ExposureTimeRaw", exposure_us)
            except Exception as e2:
                print("ExposureTimeRaw not set:", e2)


        # ---------------- Gain Auto OFF + fixed Gain --------
        try:
            # Select Analog gain channel
            gain_sel = st.PyIEnumeration(nodemap.get_node("GainSelector"))
            analog_entry = st.PyIEnumEntry(gain_sel["AnalogAll"])
            gain_sel.set_entry_value(analog_entry)

            # Turn OFF auto gain on AnalogAll
            gain_auto = st.PyIEnumeration(nodemap.get_node("GainAuto"))
            gain_auto.set_entry_value(st.PyIEnumEntry(gain_auto["Off"]))
            print("GainAuto (AnalogAll) = Off")

            # Set analog gain: 0 for brain, 6 for behavior
            gain_node = nodemap.get_node("Gain")
            if gain_node.principal_interface_type == st.EGCInterfaceType.IFloat:
                gain_val = st.PyIFloat(gain_node)
            else:
                gain_val = st.PyIInteger(gain_node)
            gain_val.value = 0 if deviceIdx == 0 else 192
            print(f"Analog Gain = {gain_val.value} (deviceIdx={deviceIdx})")

            # Optional: force DigitalAll gain to 0 as well
            if "DigitalAll" in [st.PyIEnumEntry(e).symbolic_value
                                for e in gain_sel.entries]:
                digital_entry = st.PyIEnumEntry(gain_sel["DigitalAll"])
                gain_sel.set_entry_value(digital_entry)
                gain_node = nodemap.get_node("Gain")
                if gain_node.principal_interface_type == st.EGCInterfaceType.IFloat:
                    gain_val = st.PyIFloat(gain_node)
                else:
                    gain_val = st.PyIInteger(gain_node)
                gain_val.value = 0
                print("Digital Gain = 0")
        except Exception as e:
            print("[SentechCameraStream] Gain config failed:", e)

        # ---------------- Black Level = 0 -------------------
        try:
            # Use AnalogAll selector for black level
            bl_sel = st.PyIEnumeration(nodemap.get_node("BlackLevelSelector"))
            if "AnalogAll" in [st.PyIEnumEntry(e).symbolic_value
                               for e in bl_sel.entries]:
                bl_sel.set_entry_value(st.PyIEnumEntry(bl_sel["AnalogAll"]))
            self._set_numeric(nodemap, "BlackLevel", 0)
        except Exception as e:
            print("[SentechCameraStream] BlackLevel config failed:", e)

        # ---------------- Gamma = 1.0 -----------------------
        try:
            self._set_numeric(nodemap, "Gamma", 1.0)
        except Exception as e:
            print("[SentechCameraStream] Gamma config failed:", e)


    def edit_enumeration(self, nodemap, enum_name) -> bool:
        """
        Display and edit enumeration value.

        :param nodemap: Node map.
        :param enum_name: Enumeration name.
        :return: True if enumeration value is updated. False otherwise.
        """
        node = nodemap.get_node(enum_name)
        if not node.is_writable:
            return False
        enum_node = st.PyIEnumeration(node)
        enum_entries = enum_node.entries
        print(enum_name)
        for index in range(len(enum_entries)):
            enum_entry = enum_entries[index]
            if enum_entry.is_available:
                print("{0} : {1} {2}".format(index,
                    st.PyIEnumEntry(enum_entry).symbolic_value,
                    "(Current)" if enum_node.value == enum_entry.value
                                            else ""))
        print("Else : Exit")
        selection = int(input("Select : "))
        if selection < len(enum_entries) and enum_entries[selection].is_available:
            enum_entry = enum_entries[selection]
            enum_node.set_int_value(enum_entry.value)
            return True


    def edit_setting(self, nodemap, node_name, new_value):
        """
        Edit setting which has numeric type.

        :param nodemap:  Node map.
        :param node_name: Node name.
        """
        node = nodemap.get_node(node_name)
        if not node.is_writable:
            return
        if node.principal_interface_type == st.EGCInterfaceType.IFloat:
            node_value = st.PyIFloat(node)
        elif node.principal_interface_type == st.EGCInterfaceType.IInteger:
            node_value = st.PyIInteger(node)
        while True:
            print(node_name)
            print(" Min={0} Max={1} Current={2}{3}".format(
                node_value.min, node_value.max, node_value.value,
                " Inc={0}".format(node_value.inc) if\
                        node_value.inc_mode == st.EGCIncMode.FixedIncrement\
                        else ""))

            print()
            if int(node_value.min) <= new_value <= int(node_value.max):
                node_value.value = new_value
                return


    def set_camera_side_roi(self, nodemap):
        """
        Set camera side ROI.

        :param nodemap: Node map.
        """
        region_selector = nodemap.get_node(REGION_SELECTOR)
        if not region_selector.is_writable:
            # Single ROI:
            self.edit_setting(nodemap, OFFSET_X, 80)
            self.edit_setting(nodemap, WIDTH, self.res[0])
            self.edit_setting(nodemap, OFFSET_Y, 60)
            self.edit_setting(nodemap, HEIGHT, self.res[1])
        else:
            self.edit_enumeration(nodemap, REGION_MODE)
            region_mode = st.PyIEnumeration(nodemap.get_node(REGION_MODE))
            if region_mode.current_entry.value != region_mode["Off"].value:
                self.edit_setting(nodemap, OFFSET_X, 80)
                self.edit_setting(nodemap, WIDTH, self.res[0])
                self.edit_setting(nodemap, OFFSET_Y, 60)
                self.edit_setting(nodemap, HEIGHT, self.res[1])


