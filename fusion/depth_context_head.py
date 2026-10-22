import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthContextHead(nn.Module):
    """Голова, предсказывающая распределение глубины и контекстные признаки.

    Вход:  (B, 2048, H, W)
    Выход: depth_probs      (B, num_depth_bins, H, W)  -- softmax по dim=1
           context_features (B, context_channels, H, W)
    """

    IN_CHANNELS = 2048
    HIDDEN_CHANNELS = 128

    def __init__(self, num_depth_bins: int = 80, context_channels: int = 64):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.context_channels = context_channels

        # Общая часть: 1x1 свёртка + ReLU
        self.shared = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, self.HIDDEN_CHANNELS, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # Выходная свёртка: depth + context каналов
        self.out_conv = nn.Conv2d(
            self.HIDDEN_CHANNELS, num_depth_bins + context_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        # 1. Проверка формы входа
        if x.dim() != 4:
            raise ValueError(
                f"Ожидался 4-мерный тензор (B, C, H, W), получено {x.dim()} измерений"
            )
        if x.shape[1] != self.IN_CHANNELS:
            raise ValueError(
                f"Ожидалось {self.IN_CHANNELS} каналов, получено {x.shape[1]}"
            )

        # 2. Общая часть + выходная свёртка
        out = self.out_conv(self.shared(x))

        # 3. Разделение каналов
        depth_logits, context_features = torch.split(
            out, [self.num_depth_bins, self.context_channels], dim=1
        )

        # 4. Softmax по каналам глубины: для каждой позиции (h, w)
        #    сумма по d равна 1
        depth_probs = F.softmax(depth_logits, dim=1)

        # 5. Контекстные признаки не нормализуются
        return depth_probs, context_features


def main():
    torch.manual_seed(0)

    model = DepthContextHead(num_depth_bins=80, context_channels=64)
    model.eval()

    # Искусственный вход: batch=2, 2048 каналов, карта 16x44
    x = torch.randn(2, 2048, 16, 44)

    with torch.no_grad():
        depth_probs, context_features = model(x)

    print("depth_probs shape:      ", tuple(depth_probs.shape))
    print("context_features shape: ", tuple(context_features.shape))

    # Проверка форм
    assert depth_probs.shape == (2, 80, 16, 44)
    assert context_features.shape == (2, 64, 16, 44)

    # Проверка: сумма вероятностей по каналу глубины == 1 в каждой позиции
    sums = depth_probs.sum(dim=1)
    print("sum over depth: min=%.6f max=%.6f" % (sums.min().item(), sums.max().item()))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Вероятности лежат в [0, 1]
    assert depth_probs.min() >= 0 and depth_probs.max() <= 1

    # Контекст НЕ нормализован softmax-ом (сумма не равна 1)
    assert torch.isfinite(depth_probs).all()
    assert torch.isfinite(context_features).all()

    # Проверка валидации входа
    for bad in (torch.randn(2048, 16, 44), torch.randn(2, 1024, 16, 44)):
        try:
            model(bad)
        except ValueError as e:
            print("OK, ошибка ловится:", e)
        else:
            raise AssertionError("Ожидалось ValueError")

    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()