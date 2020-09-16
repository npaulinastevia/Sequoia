import torch

from .transforms import ChannelsFirst, ChannelsLast, Compose, Transforms, ChannelsFirstIfNeeded


import pytest

@pytest.mark.parametrize("transform,input_shape,output_shape",
[   
    ## Channels first:
    (Transforms.channels_first, (9, 9, 3), (3, 9, 9)),
    # Check that the ordering doesn't get messed up:
    (Transforms.channels_first, (9, 12, 3), (3, 9, 12)),
    (Transforms.channels_first, (400, 600, 3), (3, 400, 600)),
    # Axes get permuted even when the channels are already 'first'.
    (Transforms.channels_first, (3, 12, 9), (9, 3, 12)),

    ## Channels first (if needed):
    (Transforms.channels_first_if_needed, (9, 9, 3), (3, 9, 9)),
    (Transforms.channels_first_if_needed, (9, 12, 3), (3, 9, 12)),
    (Transforms.channels_first_if_needed, (400, 600, 3), (3, 400, 600)),
    # Axes do NOT get permuted when the channels are already 'first'.
    (Transforms.channels_first_if_needed, (3, 12, 9), (3, 12, 9)),
    # Does nothing when the channel dim isn't in {1, 3}:
    (Transforms.channels_first_if_needed, (7, 12, 13), (7, 12, 13)),
    (Transforms.channels_first_if_needed, (7, 12, 123), (7, 12, 123)),
    
    ## Channels Last:
    (Transforms.channels_last, (3, 9, 9), (9, 9, 3)),
    # Check that the ordering doesn't get messed up:
    (Transforms.channels_last, (3, 9, 12), (9, 12, 3)),
    # Axes get permuted even when the channels are already 'last'.
    (Transforms.channels_last, (5, 6, 1), (6, 1, 5)),
    
    ## Channels Last (if needed):
    (Transforms.channels_last_if_needed, (3, 9, 9), (9, 9, 3)),
    # Check that the ordering doesn't get messed up:
    (Transforms.channels_last_if_needed, (3, 9, 12), (9, 12, 3)),
    # Axes do NOT get permuted when the channels are already 'last':
    (Transforms.channels_last_if_needed, (5, 6, 1), (5, 6, 1)),
    (Transforms.channels_last_if_needed, (12, 13, 3), (12, 13, 3)),
    # Does nothing when the channel dim isn't in {1, 3}:
    (Transforms.channels_last_if_needed, (7, 12, 13), (7, 12, 13)),
])
def test_transform(transform: Transforms, input_shape, output_shape):
    x = torch.rand(input_shape)
    y = transform(x)
    assert y.shape == output_shape
    assert y.shape == transform.shape_change(input_shape)


def test_compose_shape_change_same_as_result_shape():
    transform = Compose([Transforms.channels_first])
    start_shape = (9, 9, 3)
    x = transform(torch.rand(start_shape))
    assert x.shape == (3, 9, 9)
    assert x.shape == transform.shape_change(start_shape)
    assert x.shape == transform.shape_change(start_shape) == (3, 9, 9)
