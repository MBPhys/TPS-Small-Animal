import napari
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from magicgui.widgets import ComboBox, Container, LiteralEvalLineEdit, FloatSpinBox, FileEdit, SpinBox, CheckBox


from TPS_Small_Animal.utils import *
from TPS_Small_Animal.utils_metric import Gradient_NCCPRO
from TPS_Small_Animal.utils_plot import plot__regi_steps
from TPS_Small_Animal.module_Reg import Reg_model

import os 
import numpy as np
import torch
import gryds
from scipy import signal
from skimage.exposure import match_histograms



from superqt.qtcompat import QtCore
Horizontal = QtCore.Qt.Orientation.Horizontal

import subprocess
from napari.qt import thread_worker
from napari.layers import Layer
from napari.utils.notifications import show_info

import matplotlib.pyplot as plt


from itk_napari_conversion import image_from_image_layer, image_layer_from_image
import itk

class TPS(QWidget):
 """
    A widget that performs 3D-2D (CT-XRay) registration.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        A napari viewer instance.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    dict_axes_angle : dict
        A dictionary that stores the rotation angles for each axis. Initialized to {0: 0.0, 1: 0.0, 2: 0.0}.
    Image_select_CT : qtpy.QtWidgets.QWidget
        A container widget that includes a drop-down list to select a CT image.
    layer_sel_CT : napari.layers.image.Image or None
        The selected CT image layer. Initialized to None.
    Image_select_XRay : qtpy.QtWidgets.QWidget
        A container widget that includes a drop-down list to select an X-Ray image.
    layer_sel_XRay : napari.layers.image.Image or None
        The selected X-Ray image layer. Initialized to None.
    crop_modes_widget : qtpy.QtWidgets.QWidget
        A container widget that includes a drop-down list to select the projection mode. Initialized to 'MeanIP'.
    button_proc_CT : qtpy.QtWidgets.QPushButton
        A button to trigger CT preprocessing.
    button_proc_XRay : qtpy.QtWidgets.QPushButton
        A button to trigger X-Ray preprocessing.
    button_proc_initial : qtpy.QtWidgets.QPushButton
        A button to trigger fixing physical spacing.
    w_target_point : qtpy.QtWidgets.QWidget
        A container widget that includes a QLineEdit to set the target point coordinates. Initialized to (0,0,0).
    axis : qtpy.QtWidgets.QWidget
        A container widget that includes a spin box to set the rotation axis. Initialized to 0.
    angle : qtpy.QtWidgets.QWidget
        A container widget that includes a float spin box to set the rotation angle. Initialized to 0.0.
    checkbox_misal : qtpy.QtWidgets.QWidget
        A container widget that includes a checkbox to show the misalignment correction. Initialized to False.
    button_misal : qtpy.QtWidgets.QPushButton
        A button to trigger the misalignment correction.
    sim_file_Edit : qtpy.QtWidgets.QWidget
        A container widget that includes a QLineEdit to select the simulation file.
    button_sim_file : qtpy.QtWidgets.QPushButton
        A button to open the selected simulation file.
    sim_tool_file_Edit : qtpy.QtWidgets.QWidget
        A container widget that includes a QLineEdit to select the directory of the simulation tool.
    button_simulate : qtpy.QtWidgets.QPushButton
        A button to start the simulation.

    Raises
    ------
    TypeError
        If the `napari_viewer` parameter is not a `napari.Viewer` instance.

    """
    def __init__(self, napari_viewer):
        """
        Initialize the TPS widget.

        Parameters:
            napari_viewer : napari.Viewer
              A napari viewer instance.

        Raises:
            TypeError:
                If the `napari_viewer` parameter is not a `napari.Viewer` instance.
        """
        super().__init__()
        self.viewer = napari_viewer

        

        self.setLayout(QVBoxLayout())
        
        #Dictionary fo rotation angles
        self.dict_axes_angle={0 : 0.0, 1 : 0.0, 2: 0.0}
        
        #Select a CT
        possible_images_CT= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image) and x.ndim==3)]
        self.Image_select_CT=Container(widgets=[ComboBox(choices=possible_images_CT, label="Select a CT:", visible=True)])
        self.layout().addWidget(self.Image_select_CT.native)
        select_layer_name=self.Image_select_CT[0].current_choice
        try:
         self.layer_sel_CT=self.viewer.layers[select_layer_name]
         self.viewer.layers.selection.active=self.viewer.layers[select_layer_name] 
        except:
         pass   
        
        #Select a XRay
        possible_images_XRay= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image) and x.ndim==2)]
        self.Image_select_XRay=Container(widgets=[ComboBox(choices=possible_images_XRay, label="Select a X-Ray:", visible=True)])
        self.layout().addWidget(self.Image_select_XRay.native)
        select_layer_name=self.Image_select_XRay[0].current_choice
        try:
         self.layer_sel_XRay=self.viewer.layers[select_layer_name]
         self.viewer.layers.selection.active=self.viewer.layers[select_layer_name] 
        except:
         pass   
        #create mode widget
        self.crop_modes= ['MIP','MeanIP'] 
        
        self.crop_modes_widget=Container(widgets=[ComboBox(choices=self.crop_modes, label="Select projection modes:", value='MeanIP', visible=True)])
        self.layout().addWidget(self.crop_modes_widget.native)


        #if self.crop_modes_widget[0].value=='MIP':
        #   pass 
        #elif self.crop_modes_widget[0].value=='MeanIP':
        #   pass 
       
        #Preprocess CT-button and Preprocess XRay-button
        w_proc_CT_XRay = QWidget()
        w_proc_CT_XRay.setLayout(QHBoxLayout())
        self.button_proc_CT=QPushButton("Preprocess CT", self)
        self.button_proc_XRay=QPushButton("Preprocess X-Ray", self)
        w_proc_CT_XRay.layout().addWidget(self.button_proc_CT)
        w_proc_CT_XRay.layout().addWidget(self.button_proc_XRay)
        self.layout().addWidget(w_proc_CT_XRay)
        
        #w_proc_XRay = QWidget()
        #w_proc_XRay.setLayout(QHBoxLayout())
       # self.button_proc_XRay=QPushButton("Preprocess X-Ray", self)
       # w_proc_XRay.layout().addWidget(self.button_proc_XRay)
       # self.layout().addWidget(w_proc_XRay)
       
        #Match resolution and target point widget
        w_proc_initial_target = QWidget()
        w_proc_initial_target.setLayout(QHBoxLayout())
        self.button_proc_initial=QPushButton("Fix physical spacing", self)
        self.w_target_point=Container(widgets=[LiteralEvalLineEdit(value=(0,0,0), label="Target point", visible=True)])
        w_proc_initial_target.layout().addWidget(self.button_proc_initial)
        w_proc_initial_target.layout().addWidget(self.w_target_point.native)
        self.layout().addWidget(w_proc_initial_target)
    
        
        
        
        #Set up axes and angle Slider
        self.axis=Container(widgets=[SpinBox(value=0, label='Axis:', min=0, max=2)])
        self.angle=Container(widgets=[FloatSpinBox(value=0, label='Angle [°]:', min=-30, max=30, step=0.5)])
        self.w_slider_container_total = QWidget() 
        self.w_slider_container_total.setLayout(QHBoxLayout())
        self.w_slider_container_total.layout().addWidget(self.axis.native)
        self.w_slider_container_total.layout().addWidget(self.angle.native)
        self.layout().addWidget(self.w_slider_container_total)
        
        #Misalignment correction
        self.button_misal=QPushButton("Misalignment correction", self)
        self.checkbox_misal=Container(widgets=[CheckBox(value=False, label="Show correction", visible=True)])
        
        self.w_misal_total = QWidget() 
        self.w_misal_total.setLayout(QHBoxLayout())
        self.w_misal_total.layout().addWidget(self.checkbox_misal.native)
        self.w_misal_total.layout().addWidget(self.button_misal)
        self.layout().addWidget(self.w_misal_total)
        
        #Open Simulaton file
        w_sim_file_Edit = QWidget()
        w_sim_file_Edit.setLayout(QHBoxLayout())
        self.sim_file_Edit=Container(widgets=[FileEdit(label='Select the simulation file:', mode='r', filter='*.txt')])
        w_sim_file_Edit.layout().addWidget(self.sim_file_Edit.native)
        self.layout().addWidget(w_sim_file_Edit)
        
        w_sim_file = QWidget()
        w_sim_file.setLayout(QHBoxLayout())
        self.button_sim_file=QPushButton("Open Simulaton file", self)
        w_sim_file.layout().addWidget(self.button_sim_file)
        self.layout().addWidget(w_sim_file)
        
        #Simulate
        w_sim_tool_file_Edit = QWidget()
        w_sim_tool_file_Edit.setLayout(QHBoxLayout())
        self.sim_tool_file_Edit=Container(widgets=[FileEdit(label='Select the executable directory', mode='d')])
        w_sim_tool_file_Edit.layout().addWidget(self.sim_tool_file_Edit.native)
        self.layout().addWidget(w_sim_tool_file_Edit)
        
        w_simulate = QWidget()
        w_simulate.setLayout(QHBoxLayout())
        self.button_simulate=QPushButton("Simulate", self)
        w_simulate.layout().addWidget(self.button_simulate)
        self.layout().addWidget(w_simulate)
          
        
        
        
        self.Image_select_CT.changed.connect(self.selected_layer)
        self.Image_select_XRay.changed.connect(self.selected_layer)
    
        self.viewer.layers.events.inserted.connect(self.change_combo)
        self.viewer.layers.events.inserted.connect(self.remove_combo)
     
        self.viewer.layers.events.removed.connect(self.change_combo)
        self.viewer.layers.events.removed.connect(self.remove_combo)
        
        self.viewer.layers.events.inserted.connect(self.selected_layer)
        self.viewer.layers.events.removed.connect(self.selected_layer)


        self.crop_modes_widget.changed.connect(self.selected_layer)
        
        self.button_proc_initial.clicked.connect(self.match_physical_space)
        
        self.button_misal.clicked.connect(self.misalignment_correction_parallel)
        
        self.button_simulate.clicked.connect(self.execute_sim)
        self.button_sim_file.clicked.connect(self.open_sim_file)
        
        self.axis.changed.connect(self.dict_axis_change)
        self.angle.changed.connect(self.dict_angle_change)
        #self.axis.changed.connect(self.transf_3d_parallel)
        #self.angle.changed.connect(self.transf_3d_parallel)
        
        self.button_proc_CT.clicked.connect(self.preprocess_CT)
        self.button_proc_XRay.clicked.connect(self.preprocess_XRay)
        
        

        #self.button_slider_reset.clicked.connect(lambda event,   Slider_list=Slider_list: self.crop_slider_reset(event, Slider_list ))
           
    def match_physical_space(self):
    """Perform 2D-2D image spacing natching between a 2D slices (CT) and 2D (X-ray) image.

    This function matchs the physical spacing of a given X-ray image to a corresponding CT image slices using the elastix
    registration method, and adds the resulting 2D image as a new layer to the Napari viewer.

    Returns:
    None

    Raises:
    None
    """  
     # Call registration function
      possible_images= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image))] 
      self.create_proj(self.layer_sel_CT.data)
      if self.crop_modes_widget[0].value=='MIP':
           if 'MIP' in  possible_images:
              proj=self.viewer.layers['MIP']
           else:
              proj_data=np.max(self.layer_sel_CT.data, axis=0)
              data = (proj_data, {'scale': self.layer_sel_CT.scale[1:], 'translate': self.layer_sel_CT.translate[1:]}, 'image')
              proj=Layer.create(*data)
      elif self.crop_modes_widget[0].value=='MeanIP':
           if 'MeanIP' in  possible_images:
              proj=self.viewer.layers['MeanIP']
           else:    
              proj_data=np.mean(self.layer_sel_CT.data, axis=0)
              data = (proj_data, {'scale': self.layer_sel_CT.scale[1:], 'translate': self.layer_sel_CT.translate[1:]}, 'image')
              proj=Layer.create(*data)
              
      #XRay_process_data=intensity_normalize_img(self.layer_sel_XRay.data, use_nonzero_mask=False, range_norm=True)
      #XRay_process_data=1-XRay_process_data
      #XRay_process_data[XRay_process_data==1]=0    
      #XRay_tuple_Preprocess=list(self.layer_sel_XRay.as_layer_data_tuple())
      #XRay_tuple_Preprocess[0]=XRay_process_data
      #XRay_tuple_Preprocess=tuple(XRay_tuple_Preprocess)
      #XRay_layer_Preprocess=Layer.create(*XRay_tuple_Preprocess)
      
      
      fixed_image= image_from_image_layer(proj)
      moving_image= image_from_image_layer(self.layer_sel_XRay)
      fixed_image = fixed_image.astype(itk.F)
      moving_image = moving_image.astype(itk.F)
      
      dir = os.path.dirname(__file__)
      par = dir+ "/parameters-settings.txt" 
      parameter_object= itk.ParameterObject.New()
      if ".txt" in par:
          try:
             parameter_object.AddParameterFile(par)
          except:
             show_info("Parameter file not found or not valid") 
              
     
                                         
      result_image, _ = itk.elastix_registration_method(fixed_image, moving_image, parameter_object=parameter_object, log_to_console=True)
      layer = image_layer_from_image(result_image)
      layer.name =self.layer_sel_XRay.name + "_2D_registration"
      possible_images= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image))]   
      if layer.name in  possible_images:
              self.viewer.layers[layer.name].data=layer.data
      else:    
              self.viewer.add_image(layer.data,  name=layer.name, scale=layer.scale)
    
    def dict_angle_change(self):
    """
    Updates the dictionary of axis-angle pairs with the latest changes and performs
    a 3D parallel transformation based on the updated values.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
        self.dict_axes_angle[self.axis[0].value]=self.angle[0].value
        self.transf_3d_parallel()
        
    def dict_axis_change(self):
    """
    Changes the value of the first element in `self.angle` based on the corresponding value in `self.dict_axes_angle`.
    Resets the visibility of the `self.axis[0]` slider and updates the 3D transformation accordingly.

    Returns:
        None
    """
        print(self.angle[0].value)
        self.angle[0].value=self.dict_axes_angle[self.axis[0].value]
        #Set up axes and angle Slider
        self.axis[0].visible=False
        self.axis[0].visible=True
        self.transf_3d_parallel() 


    def preprocess_CT(self):
    """
    Preprocesses the selected CT image layer by performing intensity normalization and updating the `data` attribute
    of the layer with the preprocessed image.

    Parameters:
        None

    Returns:
        None
    """
        self.CT_non_preprocess_data=np.copy(self.layer_sel_CT.data)
        CT= intensity_normalize_img(self.layer_sel_CT.data, use_nonzero_mask=True, range_norm=True)
       # CT=pad_nd_image(CT, new_shape=(152, 171, 232), no_pad_left_side=False)
        self.layer_sel_CT.data=CT                            
        
        
    
    def preprocess_XRay(self):
    """
    Preprocesses the X-ray image by performing several operations:
    - pads the X-ray image to match the shape of the CT image
    - resamples the X-ray image to a fixed pixel spacing
    - normalizes the intensity of the X-ray image and inverts it
    - pads the X-ray image again to match the shape of the CT image
    - normalizes the X-ray image using the mean squared error (MSE) method
    - applies a 3x3 mean filter to the X-ray image
    - normalizes the X-ray image again using the MSE method
    - matches the histogram of the X-ray image with that of the CT image
    - sets the scale of the X-ray layer to [1,1] and updates the X-ray data with the preprocessed image

    Returns:
    None
    """
        XRay_process_data=pad_nd_image(self.layer_sel_XRay.data,  new_shape=(self.layer_sel_CT.data.shape[1],self.layer_sel_CT.data.shape[2])) 
        XRay_process_data=resample_image_2D(XRay_process_data, (0.0842, 0.0842), (0.1,0.1))
        print( XRay_process_data.shape)
        XRay_process_data=intensity_normalize_img(XRay_process_data, use_nonzero_mask=False, range_norm=True) 
        XRay_process_data=1-XRay_process_data
        XRay_process_data[XRay_process_data==1]=0
        XRay_process_data=pad_nd_image(XRay_process_data,  new_shape=(self.layer_sel_CT.data.shape[1],self.layer_sel_CT.data.shape[2]) )
        print( XRay_process_data.shape)
        print(self.layer_sel_CT.data.shape)
        #XRay_process_data=intensity_normalize_img(XRay_process_data, use_nonzero_mask=False, range_norm=True)
        XRay_process_data=normalize_minmse(XRay_process_data, pad_nd_image(np.mean(self.layer_sel_CT.data, axis=0), new_shape=XRay_process_data.shape, no_pad_left_side=False))
        XRay_process_data= signal.convolve(XRay_process_data, np.ones((3,3))/9, mode='same')
        XRay_process_data=intensity_normalize_img(XRay_process_data, use_nonzero_mask=False, range_norm=True)
        XRay_process_data=normalize_minmse(XRay_process_data,  pad_nd_image(np.mean(self.layer_sel_CT.data, axis=0), new_shape=XRay_process_data.shape, no_pad_left_side=False))
        XRay_process_data=match_histograms(XRay_process_data,  pad_nd_image(np.mean(self.layer_sel_CT.data, axis=0), new_shape=XRay_process_data.shape, no_pad_left_side=False))
        self.layer_sel_XRay.scale=[1,1]
        self.layer_sel_XRay.data=np.copy(XRay_process_data)
        
    
    
    def open_sim_file(self, event):
        """
        Opens the simulation file using Atom editor.

        Parameters:
        ----------
        event: wx.Event
            The event that triggers the function.

        Returns:
        -------
        None
        """
        atom_call= 'atom '+self.sim_file_Edit[0].value.as_posix()
        subprocess.run(atom_call, shell=True)   

    def execute_sim(self): 
    """
    Executes the simulation by calling the specified simulation tool and file (TOPAS, FLUKA, etc.).

    Returns:
        None
    """ 
        sim_call= self.sim_tool_file_Edit[0].value.as_posix() + ' ' + self.sim_file_Edit[0].value.as_posix()
        subprocess.run(sim_call, shell=True)   
    
    def transf_3d_parallel(self):
    """Applies 3D transformation to the CT volume in parallel computation.

    The function applies a 3D transformation to the CT volume using a worker thread to
    avoid blocking the main thread. Once the transformation is completed, the function
    sets the resulting volume as the data for the 'CT_Rot' layer, or adds a new layer
    with the name 'CT_Rot' and the transformed volume as data, if such layer does not
    exist yet. The function also calls the 'create_proj' method to create the projection
    image of the transformed volume.

    Returns:
        None
    """     
        def _on_done(result):
            layer_list= [x.name  for x in self.viewer.layers]
            scale_CT=self.layer_sel_CT.scale
            if 'CT_Rot' in layer_list:
               self.viewer.layers['CT_Rot'].data=result
            else:
               self.viewer.add_image(result,  name='CT_Rot', scale=scale_CT)  
            self.create_proj(result)

        
        worker =self.transf_3d()
        worker.returned.connect(_on_done)
        worker.start()
    
    
        
    
    @thread_worker    
    def transf_3d(self):
    """
     Transforms a 3D image using affine transformations based on the given rotation angles and target point.

    Parameters:
      None

    Returns:
        transformed_image_3D: ndarray
             The transformed 3D image.

    Raises:
      None 
    """
        Order=3
        target_point=self.w_target_point[0].value
        if target_point is not None and isinstance(target_point, tuple):
            target_point_l=list(target_point)
            for i, value in enumerate(target_point_l):
                target_point_l[i]=value/self.layer_sel_CT.data.shape[i]
            print(target_point_l)        
        else:
            print('Rotation center is not given')
            
        if len(target_point_l)!=3:
           target_point_l=[0,0,0]
        if not all(0<=i<=1  for i in target_point_l):
               target_point_l=[0,0,0]    
     
        axis_2=self.dict_axes_angle[2]  
        axis_1=self.dict_axes_angle[1]
        axis_0=self.dict_axes_angle[0]
        print(axis_0)
        print(axis_1)
        print(axis_2)
        transf_3D = gryds.AffineTransformation(ndim=3, angles=[angletorad(axis_0),angletorad(axis_1), angletorad(axis_2)], 
                    translation=[0,0, 0],  center=target_point_l)
        interpolator = gryds.Interpolator(self.layer_sel_CT.data)
        transformed_image_3D = interpolator.transform(transf_3D, order=Order)
        return transformed_image_3D 
    
    def misalignment_correction_parallel(self):
    """Perform misalignment correction in parallel.

    This function uses the `misalignment_correction` method to correct misalignments between a 2D image stack and a 3D volume. The correction is performed in parallel, and the results are displayed in the  
    viewer window.

    Returns:
        None

    Raises:
        None
    """
        def _on_done(result):
            mov_2D, mov_3D =result
            self.viewer.add_image(mov_2D)
            self.viewer.add_image(mov_3D)
        def _plot_regi(values):             
             #Set figure
             self.fig = plt.figure() #figsize=(15, 9)
             mov, proj_init_numpy0, tar, gradncc_sim_list, angle_0_list, angle_1_list, angle_2_list, trans_1_list, trans_2_list = values
             self.canvas_fig_new= plot__regi_steps(self.fig, self.viewer, mov, proj_init_numpy0, tar, gradncc_sim_list, angle_0_list, angle_1_list, angle_2_list, trans_1_list, trans_2_list)
             if self.i % 10 == 0:
              try: 
               self.viewer.window.remove_dock_widget(self.canvas_fig)
              except:
               pass

              self.viewer.window.add_dock_widget(self.canvas_fig_new, area='top')
              self.canvas_fig=self.canvas_fig_new
             self.i=self.i+1 
        
        self.i=0
        worker =self.misalignment_correction()
        worker.returned.connect(_on_done)
        if self.checkbox_misal[0].value==True:
           worker.yielded.connect(_plot_regi)
        else:
           worker.yielded.disconnect 
        worker.start()
    
    
        
    
    
    
    @thread_worker    
    def misalignment_correction(self):
    """
    Apply misalignment correction to a 3D image (CT) and 2D projection (X-Ray) using a projection model to produce a DRR and gradient-based optimization.

    Parameters:
        self : object
               An instance of the class that calls this function.

    Returns:
        tuple: (np.ndarray: Aligned projection of the DRR to 2D target projection (X-Ray), np.ndarray: Aligned DRR to 2D target projection (X-Ray)) 
         
         

    Notes
    -----
    This function uses a projection model to align a 3D image (CT) and 2D projection (X-Ray) by optimizing a gradient-based loss function. It takes the following steps:

    1. Sets the initial translation and angles.
    2. Sets the center of rotation point.
    3. Sets the optimizer and loss function.
    4. Runs the training loop strategy, which includes updating the rotation angles and translation based on the optimizer, checking for convergence, and breaking if the convergence threshold is met.
    5. Creates a list of angles and translations for plotting.
    6. Uses the projection model to create a 3D projection of the moving image (DRR).
    7. Computes the gradient of normalized cross correlation (NCC) loss between the target (X-Ray) and moving image (DRR).
    8. Backpropagates the loss to update the rotation angles and translation.
    9. Updates the optimizer to zero and appends the gradient NCC loss to a list for plotting.

    """        
        device = torch.device("cuda")
        
        #Set initial translation and angles
        angle_0 = torch.tensor([[0.0]], dtype=torch.float, requires_grad=False, device=device)
        angle_1 = torch.tensor([[0.0]], dtype=torch.float, requires_grad=False, device=device)
        angle_2 = torch.tensor([[0.0]], dtype=torch.float, requires_grad=False, device=device)
        trans_1 = torch.tensor([[0.0]], dtype=torch.float, requires_grad=False, device=device)
        trans_2 = torch.tensor([[0.0]], dtype=torch.float, requires_grad=False, device=device)
        
        #Set center of rotation point
        target_point=self.w_target_point[0].value
        if target_point is not None and isinstance(target_point, tuple):
            target_point_l=list(target_point)
            for i, value in enumerate(target_point_l):
                target_point_l[i]=value
            print(target_point_l)        
        else:
            print('Rotation center is not given')
        if len(target_point_l)!=3:
           target_point_l=[0,0,0]
        if not all(0<=i<=1  for i in target_point_l):
               target_point_l=[0,0,0]   
        center_point = torch.tensor([target_point_l], dtype=torch.float, requires_grad=False, device=device)
        
        #Set optimizer and loss function
        optimizer_gradncc = torch.optim.Adam([angle_0, angle_1, angle_2, trans_1, trans_2],  lr=0.01)
        criterion_gradncc = Gradient_NCCPRO()
        
        # training loop strategy
        angle_0_list=[]
        angle_1_list=[]
        angle_2_list=[]
        trans_1_list=[]
        trans_2_list=[]
        gradncc_sim_list = []
        switch_angle_1=False
        switch_angle_12=False
        switch_all=False
        learning_rate=False
        stop=False
        ITER_STEPS = 500
        switch_trd = 0.001
        stop_trd = 0.000001
        iter_value= ITER_STEPS
        
        model= Reg_model().to(device)
        CT_tensor=tensor_exp2torch(self.layer_sel_CT.data, 1, device)
        
        possible_images= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image))]   
        if self.layer_sel_XRay.name +"_2D_registration" in  possible_images:
           print('Test') 
           target_tensor = tensor_exp2torch(self.viewer.layers[self.layer_sel_XRay.name +"_2D_registration"].data, 1, device) 
        else:    
           target_tensor = tensor_exp2torch(self.layer_sel_XRay.data, 1, device)
        
        for iter in range(ITER_STEPS):
         iter_difference=iter-iter_value 
         print(iter)
         gradncc_sim_list_np = np.array(gradncc_sim_list)
         if iter >10: 
           print(np.std(gradncc_sim_list_np[-10:]))
         angle_0.requires_grad=True 
         if iter > 10 and angle_1.requires_grad==False and switch_angle_1==False:
                if np.std(gradncc_sim_list_np[-10:]) < switch_trd:
                   angle_0.requires_grad=False  
                   angle_1.requires_grad=True
                   switch_angle_1=True
                   iter_value=np.copy(iter)
         elif iter_difference> 20 and switch_angle_1==True:
                if np.std(gradncc_sim_list_np[-10:]) < switch_trd:
                   angle_0.requires_grad=True  
                   angle_1.requires_grad=True 
                   switch_angle_12=True
                   switch_angle_1=False
                   iter_value=np.copy(iter)
         elif iter_difference> 20 and switch_angle_12==True:
             if np.std(gradncc_sim_list_np[-10:]) < switch_trd:
                           angle_0.requires_grad=True  
                           angle_1.requires_grad=True  
                           angle_2.requires_grad=True  
                           trans_1.requires_grad=True  
                           trans_2.requires_grad=True
                           switch_angle_12=False
                           learning_rate=True
                           switch_all=True
                           iter_value=np.copy(iter)
         elif iter_difference> 50 and switch_all==True and learning_rate==True:
             optimizer_gradncc.param_groups[0]['lr'] = 0.001
             iter_value=np.copy(iter)
             learning_rate=False
                   
         elif iter_difference> 50 and switch_all==True and learning_rate==False:
            stop = np.std(gradncc_sim_list_np[-10:]) < stop_trd
        
         if stop:
          break 
      
        # Set optimizer_grad to zero
         optimizer_gradncc.zero_grad()   
        
        #Create list for plot
         angle_0_list.append(angle_0.detach().cpu().numpy())
         angle_1_list.append(angle_1.detach().cpu().numpy())
         angle_2_list.append(angle_2.detach().cpu().numpy())
         trans_1_list.append(trans_1.detach().cpu().numpy())
         trans_2_list.append(trans_2.detach().cpu().numpy())
        
        #Projextion model
         mov, mov_3D, tar = model(CT_tensor, target_tensor, angle_0, angle_1, angle_2, trans_1, trans_2, center_point)   
        
        # Backward graph   
         gradncc_loss = criterion_gradncc(target_tensor, mov)
         gradncc_sim_list.append(gradncc_loss.item())
         gradncc_loss.backward()
         optimizer_gradncc.step()
        
         if iter == 0:
            proj_init_numpy0 = np.array(mov[0,0,:,:].data.cpu())
         
         if self.checkbox_misal[0].value==True:
            yielded_parameters=(mov, proj_init_numpy0, tar, gradncc_sim_list, angle_0_list, angle_1_list, angle_2_list, trans_1_list, trans_2_list) 
            yield yielded_parameters
            
        if self.CT_non_preprocess_data is not None:   
          CT_tensor=tensor_exp2torch(self.CT_non_preprocess_data, 1, device)
          mov, mov_3D, tar = model(CT_tensor, target_tensor, angle_0, angle_1, angle_2, trans_1, trans_2, center_point)   
        
        return (np.array((mov[0,0,:,:]).data.cpu()), np.array((mov_3D[0,0,:,:,:]).data.cpu()))
        
    def create_proj(self, img_3D):
        """Create a projection of a 3D image and add it to the viewer.

        Parameters:
           img_3D (ndarray): A 3D image to project.

        Returns:
           None

        Notes:
          - The projection type is determined by the value of the `crop_modes_widget` attribute.
          - The projection image is added to the viewer as a new layer, or updated if the layer already exists.
        """
        scale_proj=self.layer_sel_CT.scale[1:3]
        possible_images= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image))]   
        if self.crop_modes_widget[0].value=='MIP':
           max_proj_img=self.max_proj(img_3D)
           if 'MIP' in  possible_images:
              self.viewer.layers['MIP'].data=max_proj_img
           else:    
              self.viewer.add_image(max_proj_img,  name='MIP', scale=scale_proj)
        elif self.crop_modes_widget[0].value=='MeanIP':
           mean_proj_img=self.mean_proj(img_3D)
           if 'MeanIP' in  possible_images:
              self.viewer.layers['MeanIP'].data=mean_proj_img
           else:    
              self.viewer.add_image(mean_proj_img,  name='MeanIP', scale=scale_proj)
          
    
    def max_proj(self, img, slices_num=1000):
    """Compute the maximum intensity projection (MIP) of a 3D image along the Z axis.

    Parameters:
        img (numpy.ndarray): A 3D image array of shape (z, y, x).
        slices_num (int): The number of slices to include in the MIP calculation. Defaults to 1000.

    Returns:
        numpy.ndarray: A 2D MIP image array of shape (y, x).
    """
        img_shape = img.shape     
        for i in range(img_shape[0]):
            start = max(0, i-slices_num)
            mip = np.amax(img[start:i+1],0)
        return mip
    
    
    def mean_proj(self, img, slices_num=1000):
    """
    Compute the mean projection of a 3D numpy array along the z-axis.

    Parameters:
        img (np.ndarray): A 3D numpy array containing the input image data.
        slices_num (int, optional): The number of slices to average. Defaults to 1000.

    Returns:
        np.ndarray: A 2D numpy array representing the mean projection of the input image along the z-axis.

    """
        img_shape = img.shape
        for i in range(img_shape[0]):
            start = max(0, i-slices_num)
            mean= np.mean(img[start:i+1],0)
        return mean
   
    def selected_layer(self):
    """Selects the layer for CT and X-ray images.

    This function sets the selected CT and X-ray image layers based on the current 
    user selection. It tries to select the specified layer and, if successful, sets it 
    as the active layer. If the selection fails, it does nothing.

    Parameters:
       self : object
        Object instance that holds the Image_select_CT and Image_select_XRay lists and 
        viewer layers.

    Returns:
       None
    """
            select_layer_name=self.Image_select_CT[0].current_choice
            try:
             self.layer_sel_CT=self.viewer.layers[select_layer_name] 
             self.viewer.layers.selection.active=self.viewer.layers[select_layer_name] 
            except:
             pass   
            select_layer_name=self.Image_select_XRay[0].current_choice
            try:
             self.layer_sel_XRay=self.viewer.layers[select_layer_name] 
             self.viewer.layers.selection.active=self.viewer.layers[select_layer_name] 
            except:
             pass   



    def remove_combo(self, event):
    """
    Removes the first two widgets in the layout of the current instance of 
    `QComboBox` when called in response to the given `event`.

    Parameters:
      event : QEvent
         The event that triggers the removal of the widgets.

    Returns:

    None
    """
            self.layout().itemAt(0).widget().deleteLater()
            self.layout().itemAt(1).widget().deleteLater()


             
    def change_combo(self, event):
    """
    Callback function for updating the CT and X-Ray image selection.

    Parameters:
       event : napari.utils.event.Event
         The event object.

    Returns:
       None

    Notes
    -----
    This function updates the CT and X-Ray image selection drop-down menus based on the available image layers in the
    viewer. It also updates the current choice for each menu if it was previously selected. Finally, it connects the
    change event of each menu to the `selected_layer` function.

    """

             possible_images_CT= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image) and x.ndim==3)]
             self.CT_non_preprocess_data=None
             Image_select_old=self.Image_select_CT
             try:
              self.Image_select_CT=Container(widgets=[ComboBox(choices=possible_images_CT, label="Select a CT", value=Image_select_old[0].current_choice)])

              self.layout().insertWidget(0,self.Image_select_CT.native )
             except:
              self.Image_select_CT=Container(widgets=[ComboBox(choices=possible_images_CT, label="Select a CT")])
              self.layout().insertWidget(0,self.Image_select_CT.native ) 
             
             self.Image_select_CT.changed.connect(self.selected_layer)
             
             possible_images_XRay= [x.name  for x in self.viewer.layers if (isinstance(x, napari.layers.image.image.Image) and x.ndim==2)]

             Image_select_old=self.Image_select_XRay
             try:
              self.Image_select_XRay=Container(widgets=[ComboBox(choices=possible_images_XRay, label="Select a X-Ray", value=Image_select_old[0].current_choice)])

              self.layout().insertWidget(1,self.Image_select_XRay.native )
             except:
              self.Image_select_XRay=Container(widgets=[ComboBox(choices=possible_images_XRay, label="Select a X-Ray")])
              self.layout().insertWidget(1,self.Image_select_XRay.native ) 

             self.Image_select_XRay.changed.connect(self.selected_layer)
             
             self.selected_layer()




@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [TPS]
