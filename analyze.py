""" Example script on how to start matlab engine instance
and interwave spectral python and matlab hyperspectral image
toolbox source code.

Requires valid matlab license instance. For additional
requirements see requirements.txt

Spectral function just shows the some example data manipulation using SPy
"""
import matlab.engine
import spectral.io.envi as envi
import torch

def run_engine():
    # Start matlab engine
    core = matlab.engine.start_matlab()

    # Sample data
    # Would have included in repo, but github has size limits.
    # See https://gitlab.citius.gal/hiperespectral/ChangeDetectionDataset
    first = "BayArea2013PV.raw"
    second = "BayArea2015PV.raw"

    # Example code on how to detect change overtime for a given
    # hyperspectral image raw data set using MATLAB hyperspectral imaging toolbox.
    matlab_code = f"""

    datacube_2013 = hypercube({first})
    datacube_2015 = hypercube({second})
    windowSize = 7;

    changeMap(datacube_2013, datacube_2015, windowSize)

    function changeMap = changeDetection(imageData1, imageData2, windowSize)
        % Get the center of window.
        centerPixel = ((windowSize-1)/2);

        % Get the size of the input data.
        [row,col,~] = size(imageData1);

        if isinteger(imageData1)
            imageData1 = single(imageData1);
            imageData2 = single(imageData2);
        end

        % Apply zero padding to handle the edge pixels.
        imageData1 = padarray(imageData1,[centerPixel centerPixel],0,"both");
        imageData2 = padarray(imageData2,[centerPixel centerPixel],0,"both");

        % Initialize the change map output.
        changeMap = zeros(row,col);

        for r = (centerPixel + 1):(row + centerPixel)
            for c = (centerPixel + 1):(col + centerPixel)
                rowNeighborhood = (r - centerPixel):(r + centerPixel);
                colNeighborhood = (c - centerPixel):(c + centerPixel);
                % Find the Euclidean distance between the reference signature and
                % the neighborhood of the target signature.
                spectra1 = reshape(imageData1(r,c,:),1,[]);
                spectra2 = reshape(imageData2(rowNeighborhood,colNeighborhood,:), ...
                    windowSize*windowSize,size(imageData1,3));
                a = min(pdist2(spectra1,spectra2));
                % Find the Euclidean distance between the target signature and
                % the neighborhood of the reference signature.
                spectra1 = reshape(imageData2(r,c,:),1,[]);
                spectra2 = reshape(imageData1(rowNeighborhood,colNeighborhood,:), ...
                    windowSize*windowSize,size(imageData1,3));
                b = min(pdist2(spectra1, spectra2));
                % Store the pixel-wise results in the change map.
                changeMap(r - centerPixel,c - centerPixel) = max(a,b);
            end
        end
    end
    """
    core.eval(matlab_code, nargout=0)

def spectral():
    """Example of using multi/hyperspectral data
    """
    # Find Indian Pine dataset here: https://purr.purdue.edu/publications/1947/1
    # Used gdal from osgeo to create an hdr version for envi analysis.
    data = envi.open('19920612_AVIRIS_IndianPine_Site3.hdr', '19920612_AVIRIS_IndianPine_Site3.tif')
    print(f"Metadata for Indian Pine dataset: {data}")

    # Memmap version of data
    # Good for very large data set, as only parts are loaded into memory
    memmap = data.open_memmap()

    # Dictionary of hyperspectral band data
    bands = {}
    for band in range(data.nbands):
        bands[band] = data.read_band(band)

    print(f"Example {type(bands[1])} dictionary contents: {bands[1]}")
    # PyTorch tensor of the mmmap data
    tensor = torch.Tensor(memmap.copy())
    print(f" Example tensor histogram: {tensor.histogram()}")

