
# Signal Mixer

## Overview

The Signal Mixer is a desktop application designed and implemented using PyQt5. It provides an interactive environment to explore the relative importance of magnitude and phase components in signals, using grayscale images as a visual representation. The project emphasizes the understanding of frequency contributions within the signal through a variety of features.

## Features

### Image Viewers

- **Open and View Images**: Load and view up to four grayscale images concurrently in separate viewports.
  - **Color Conversion**: Automatically converts colored images to grayscale for consistency.
  - **Unified Size**: Ensures all images are displayed at the size of the smallest image.

### Browse Functionality

- **Easy Image Switching**: Double-click on an image viewer to seamlessly switch to another image.

### Output Handling

- **Two Output Ports**: Visualize the mixer result in one of the two output viewports, mirroring the input image viewport.

### Image Adjustment

- **Brightness/Contrast Control**: Dynamically adjust image brightness and contrast using mouse dragging in any image viewport.

### Components Mixer

- **Weighted Average Calculation**: Generate output images by performing an Inverse Fourier Transform (ifft) of the weighted average of Fourier Transform (FT) components from the input images.
- **Customizable Weights**: Intuitively customize weights for each image's FT using sliders.

### Regions Mixer

- **Region Selection**: Choose regions for each FT component (inner for low frequencies or outer for high frequencies).
- **Customizable Region Size**: Adjust the size or percentage of the selected region using sliders or resize handles.
- **Unified Region Selection**: Ensure consistency in region selection across all four images.

### Real-time Mixing

- **Operation Handling**: Automatically cancels the previous operation if the user initiates a new request while the previous one is still running.
## Contributers

<table>
  <tr>
   <td align="center">
      <a href="https://github.com/Nadaaomran">
        <img src="https://avatars.githubusercontent.com/u/104179154?v=4" width="100px;" alt="Nada"/>
        <br />
        <sub><b>Nada Omran</b></sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/hadeerfasih">
        <img src="https://avatars.githubusercontent.com/u/104545742?v=4" width="100px;" alt="Hadeer"/>
        <br />
        <sub><b>Hadeer Fasih</b></sub>
      </a>
      <br />
    </td>
   <td align="center">
      <a href="https://github.com/Mariam-Hatem">
        <img src="https://avatars.githubusercontent.com/u/115348754?v=4" width="100px;" alt="Mariam"/>
        <br />
        <sub><b>Mariam Hatem</b></sub>
      </a>
      <br />

  </tr>
</table>

