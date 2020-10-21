import torch
import NCC



images = torch.randn(4, 3, 15, 15)  # 4 images, 3 channels, 15x15 pixels each
patch_shape = 3, 5, 5  # 3 channels, 5x5 pixels neighborhood
stds = NCC.patch_std(images, patch_shape)
patch = images[3, 2, :5, :5]
expected_std = patch.std(unbiased=False)  # standard deviation of the third image, channel 2, top left 5x5 patch
computed_std = stds[3, 2, 5 // 2, 5 // 2]  # computed standard deviation whose 5x5 neighborhood covers same patch
computed_std.isclose(expected_std).item()

print(expected_std)
print(computed_std)
print(computed_std.isclose(expected_std).item())