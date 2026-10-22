"""Дискретные интервалы глубины для операции Lift в LSS.

Диапазон [d_min, d_max] делится на num_bins равных интервалов.
Для num_bins интервалов нужно num_bins + 1 границ.
"""
from typing import Tuple

import torch

DEFAULT_D_MIN = 1.0     # метры
DEFAULT_D_MAX = 60.0    # метры
DEFAULT_NUM_BINS = 80   # должно совпадать с num_depth_bins в DepthContextHead


def create_depth_bins(
    d_min: float = DEFAULT_D_MIN,
    d_max: float = DEFAULT_D_MAX,
    num_bins: int = DEFAULT_NUM_BINS,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Строит равномерные интервалы глубины.

    Возвращает:
        edges   (num_bins + 1,) -- границы интервалов, от d_min до d_max
        centers (num_bins,)     -- центры интервалов (физическая глубина бина)
        width   float           -- ширина одного интервала, (d_max - d_min) / num_bins
    """
    if num_bins < 1:
        raise ValueError(f"num_bins должно быть >= 1, получено {num_bins}")
    if d_min <= 0:
        raise ValueError(f"d_min должно быть > 0 (глубина положительна), получено {d_min}")
    if d_max <= d_min:
        raise ValueError(f"d_max ({d_max}) должно быть больше d_min ({d_min})")

    width = (d_max - d_min) / num_bins
    edges = torch.linspace(d_min, d_max, num_bins + 1, dtype=torch.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, width


def validate_depth_bins(
    edges: torch.Tensor,
    centers: torch.Tensor,
    d_min: float = DEFAULT_D_MIN,
    d_max: float = DEFAULT_D_MAX,
    num_bins: int = DEFAULT_NUM_BINS,
    atol: float = 1e-5,
) -> None:
    """Проверяет корректность границ и центров. При нарушении бросает ValueError."""
    if edges.shape != (num_bins + 1,):
        raise ValueError(f"edges: ожидалась форма ({num_bins + 1},), получено {tuple(edges.shape)}")
    if centers.shape != (num_bins,):
        raise ValueError(f"centers: ожидалась форма ({num_bins},), получено {tuple(centers.shape)}")
    if not torch.isfinite(edges).all():
        raise ValueError("Границы содержат NaN или бесконечность")

    if not torch.isfinite(centers).all():
        raise ValueError("Центры содержат NaN или бесконечность")
    
    if abs(edges[0].item() - d_min) > atol or abs(edges[-1].item() - d_max) > atol:
        raise ValueError(
            f"Границы выходят за диапазон: [{edges[0].item():.4f}, {edges[-1].item():.4f}] "
            f"вместо [{d_min}, {d_max}]"
        )

    widths = edges[1:] - edges[:-1]
    if not (widths > 0).all():
        raise ValueError("Границы должны строго возрастать")

    expected_width = (d_max - d_min) / num_bins
    if not torch.allclose(widths, torch.full_like(widths, expected_width), atol=atol):
        raise ValueError("Интервалы неравномерны")

    if not ((centers > edges[:-1]) & (centers < edges[1:])).all():
        raise ValueError("Каждый центр должен лежать строго внутри своего интервала")
    
    expected_centers = 0.5 * (edges[:-1] + edges[1:])

    if not torch.allclose(
        centers,
        expected_centers,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("Центры должны совпадать с серединами интервалов")


def main():
    edges, centers, width = create_depth_bins()
    validate_depth_bins(edges, centers)

    print(f"Число границ:   {edges.numel()}")
    print(f"Число центров:  {centers.numel()}")
    print(f"Ширина бина:    {width:.4f} м")
    print(f"Первые границы: {edges[:3].tolist()}")
    print(f"Последние:      {edges[-3:].tolist()}")
    print(f"Первый центр:   {centers[0].item():.4f} м")
    print(f"Последний центр:{centers[-1].item():.4f} м")

    for bad in (dict(d_min=0.0), dict(d_max=0.5), dict(num_bins=0)):
        try:
            create_depth_bins(**bad)
        except ValueError as e:
            print("OK, ошибка ловится:", e)
        else:
            raise AssertionError(f"Ожидалось ValueError для {bad}")

    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
