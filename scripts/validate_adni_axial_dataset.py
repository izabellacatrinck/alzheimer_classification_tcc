from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SliceValidation:
    stem: str
    npy_path: Path | None
    png_path: Path | None
    npy_ok: bool
    png_ok: bool
    array_shape: tuple[int, ...] | None
    dtype: str | None
    min_value: float | None
    max_value: float | None
    issues: list[str]


@dataclass
class PatientValidation:
    patient_id: str
    axial_dir: Path
    slice_results: list[SliceValidation]
    issues: list[str]
    contact_sheet_path: Path | None

    @property
    def status(self) -> str:
        return "OK" if not self.issues and all(not s.issues for s in self.slice_results) else "ERROR"

    @property
    def npy_count(self) -> int:
        return sum(1 for s in self.slice_results if s.npy_path is not None)

    @property
    def png_count(self) -> int:
        return sum(1 for s in self.slice_results if s.png_path is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida as fatias axiais do dataset ADNI_PROCESSADO e gera "
            "pranchas visuais por paciente a partir dos arquivos NPY."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data") / "ADNI_PROCESSADO_4",
        help="Diretorio raiz do dataset processado.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data") / "validacao_adni_axial",
        help="Diretorio onde o relatorio e as imagens de validacao serao salvos.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=5,
        help="Numero de colunas da prancha visual por paciente.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolucao das pranchas visuais.",
    )
    return parser.parse_args()


def discover_axial_dirs(data_dir: Path) -> list[Path]:
    patient_dirs = [path for path in sorted(data_dir.iterdir()) if path.is_dir()]
    return [patient_dir / "slices_entropy_axial" for patient_dir in patient_dirs if (patient_dir / "slices_entropy_axial").is_dir()]


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(f"Esperado array 2D, encontrado shape {image.shape}")

    finite_mask = np.isfinite(image)
    if not finite_mask.all():
        image = np.where(finite_mask, image, 0.0)

    min_value = float(image.min())
    max_value = float(image.max())

    if math.isclose(min_value, max_value):
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_value) / (max_value - min_value)


def validate_png(png_path: Path) -> tuple[bool, str | None]:
    try:
        with png_path.open("rb") as fp:
            signature = fp.read(8)
        expected_signature = b"\x89PNG\r\n\x1a\n"
        if signature != expected_signature:
            return False, "assinatura PNG invalida"
        return True, None
    except OSError as exc:
        return False, f"erro ao abrir PNG: {exc}"


def validate_slice(stem: str, npy_path: Path | None, png_path: Path | None) -> SliceValidation:
    issues: list[str] = []
    npy_ok = False
    png_ok = False
    array_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    min_value: float | None = None
    max_value: float | None = None

    if npy_path is None:
        issues.append("arquivo NPY ausente")
    else:
        try:
            array = np.load(npy_path)
            array_shape = tuple(int(dim) for dim in array.shape)
            dtype = str(array.dtype)
            min_value = float(np.nanmin(array))
            max_value = float(np.nanmax(array))

            if array.ndim != 2:
                issues.append(f"NPY nao e 2D: shape={array_shape}")
            if not np.isfinite(array).all():
                issues.append("NPY contem NaN ou infinito")

            npy_ok = not any(issue.startswith("NPY") or "NaN" in issue for issue in issues)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"erro ao carregar NPY: {exc}")

    if png_path is None:
        issues.append("arquivo PNG ausente")
    else:
        png_ok, png_issue = validate_png(png_path)
        if png_issue:
            issues.append(png_issue)

    return SliceValidation(
        stem=stem,
        npy_path=npy_path,
        png_path=png_path,
        npy_ok=npy_ok,
        png_ok=png_ok,
        array_shape=array_shape,
        dtype=dtype,
        min_value=min_value,
        max_value=max_value,
        issues=issues,
    )


def collect_slice_results(axial_dir: Path) -> list[SliceValidation]:
    npy_files = {path.stem: path for path in sorted(axial_dir.glob("*.npy"))}
    png_files = {path.stem: path for path in sorted(axial_dir.glob("*.png"))}
    all_stems = sorted(set(npy_files) | set(png_files))
    return [validate_slice(stem, npy_files.get(stem), png_files.get(stem)) for stem in all_stems]


def create_contact_sheet(
    patient_id: str,
    slice_results: list[SliceValidation],
    output_path: Path,
    columns: int,
    dpi: int,
) -> None:
    valid_results = [result for result in slice_results if result.npy_path and result.npy_ok]
    if not valid_results:
        logger.info("Nenhuma fatia valida disponível para gerar prancha do paciente %s", patient_id)
        return

    rows = math.ceil(len(valid_results) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.6, rows * 2.8), dpi=dpi)
    axes_array = np.atleast_1d(axes).ravel()

    for ax in axes_array:
        ax.axis("off")

    for ax, result in zip(axes_array, valid_results):
        array = np.load(result.npy_path)
        normalized = normalize_for_display(array)
        ax.imshow(normalized, cmap="gray")
        ax.set_title(result.stem.split("_")[-1], fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{patient_id} - axial ({len(valid_results)} fatias)", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prancha visual gerada para %s em %s", patient_id, output_path)


def validate_patient(axial_dir: Path, output_dir: Path, columns: int, dpi: int) -> PatientValidation:
    patient_id = axial_dir.parent.name
    logger.info("Iniciando validacao do paciente %s", patient_id)
    slice_results = collect_slice_results(axial_dir)
    patient_issues: list[str] = []

    if not slice_results:
        patient_issues.append("nenhuma fatia encontrada")

    expected_count = len(slice_results)
    if sum(1 for result in slice_results if result.npy_path is not None) != expected_count:
        patient_issues.append("ha fatias sem NPY correspondente")
    if sum(1 for result in slice_results if result.png_path is not None) != expected_count:
        patient_issues.append("ha fatias sem PNG correspondente")

    contact_sheet_path = output_dir / "contact_sheets" / f"{patient_id}_axial_contact_sheet.png"
    create_contact_sheet(patient_id, slice_results, contact_sheet_path, columns, dpi)

    if not contact_sheet_path.exists():
        contact_sheet_path = None
        patient_issues.append("nao foi possivel gerar a prancha visual")

    validation = PatientValidation(
        patient_id=patient_id,
        axial_dir=axial_dir,
        slice_results=slice_results,
        issues=patient_issues,
        contact_sheet_path=contact_sheet_path,
    )

    if validation.status == "ERROR":
        logger.warning(
            "Paciente %s validado com erros (%d problemas de fatia, %s)",
            patient_id,
            sum(1 for result in slice_results if result.issues),
            "; ".join(validation.issues) or "problemas encontrados",
        )
    else:
        logger.info("Paciente %s validado com sucesso", patient_id)

    return validation


def write_summary_csv(results: Iterable[PatientValidation], output_path: Path) -> None:
    logger.info("Escrevendo resumo CSV em %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "patient_id",
                "status",
                "axial_dir",
                "npy_count",
                "png_count",
                "patient_issues",
                "slice_issues_count",
                "contact_sheet",
            ]
        )
        for result in results:
            slice_issue_count = sum(1 for slice_result in result.slice_results if slice_result.issues)
            writer.writerow(
                [
                    result.patient_id,
                    result.status,
                    str(result.axial_dir),
                    result.npy_count,
                    result.png_count,
                    " | ".join(result.issues),
                    slice_issue_count,
                    str(result.contact_sheet_path) if result.contact_sheet_path else "",
                ]
            )


def write_slice_details_csv(results: Iterable[PatientValidation], output_path: Path) -> None:
    logger.info("Escrevendo detalhes por fatia em %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "patient_id",
                "slice_stem",
                "npy_ok",
                "png_ok",
                "shape",
                "dtype",
                "min_value",
                "max_value",
                "issues",
                "npy_path",
                "png_path",
            ]
        )
        for result in results:
            for slice_result in result.slice_results:
                writer.writerow(
                    [
                        result.patient_id,
                        slice_result.stem,
                        slice_result.npy_ok,
                        slice_result.png_ok,
                        slice_result.array_shape,
                        slice_result.dtype,
                        slice_result.min_value,
                        slice_result.max_value,
                        " | ".join(slice_result.issues),
                        str(slice_result.npy_path) if slice_result.npy_path else "",
                        str(slice_result.png_path) if slice_result.png_path else "",
                    ]
                )


def write_html_report(results: list[PatientValidation], output_path: Path) -> None:
    logger.info("Escrevendo relatorio HTML em %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_patients = len(results)
    ok_patients = sum(1 for result in results if result.status == "OK")
    error_patients = total_patients - ok_patients

    cards: list[str] = []
    for result in results:
        issues = result.issues + [
            f"{slice_result.stem}: {'; '.join(slice_result.issues)}"
            for slice_result in result.slice_results
            if slice_result.issues
        ]
        issue_html = "<br>".join(issues) if issues else "Sem problemas encontrados."
        contact_sheet_html = (
            f'<img src="{result.contact_sheet_path.relative_to(output_path.parent).as_posix()}" '
            f'alt="Prancha visual de {result.patient_id}">'
            if result.contact_sheet_path
            else "<p>Prancha nao gerada.</p>"
        )
        cards.append(
            f"""
            <section class="card">
                <div class="card-header">
                    <h2>{result.patient_id}</h2>
                    <span class="status {result.status.lower()}">{result.status}</span>
                </div>
                <p><strong>Axial:</strong> {result.axial_dir}</p>
                <p><strong>NPY:</strong> {result.npy_count} | <strong>PNG:</strong> {result.png_count}</p>
                <p><strong>Problemas:</strong><br>{issue_html}</p>
                {contact_sheet_html}
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validacao ADNI Processado - Axial</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 24px;
            background: #f5f7fb;
            color: #1f2937;
        }}
        h1 {{
            margin-top: 0;
        }}
        .summary {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }}
        .summary-card, .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            padding: 16px;
        }}
        .summary-card {{
            min-width: 180px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 20px;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }}
        .status {{
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status.ok {{
            background: #dcfce7;
            color: #166534;
        }}
        .status.error {{
            background: #fee2e2;
            color: #991b1b;
        }}
        img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            background: #111827;
        }}
        strong {{
            color: #111827;
        }}
    </style>
</head>
<body>
    <h1>Validacao visual do dataset ADNI_PROCESSADO (axial)</h1>
    <div class="summary">
        <div class="summary-card"><strong>Pacientes</strong><br>{total_patients}</div>
        <div class="summary-card"><strong>OK</strong><br>{ok_patients}</div>
        <div class="summary-card"><strong>Com erro</strong><br>{error_patients}</div>
    </div>
    <div class="grid">
        {''.join(cards)}
    </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {data_dir}")

    logger.info("Iniciando validacao do dataset ADNI_PROCESSADO")
    logger.info("Diretorio de dados: %s", data_dir)
    logger.info("Diretorio de saida: %s", output_dir)

    axial_dirs = discover_axial_dirs(data_dir)
    if not axial_dirs:
        raise FileNotFoundError(f"Nenhuma pasta axial foi encontrada em: {data_dir}")

    logger.info("%d pastas axiais encontradas", len(axial_dirs))

    results: list[PatientValidation] = []
    for axial_dir in axial_dirs:
        results.append(validate_patient(axial_dir, output_dir, args.columns, args.dpi))

    write_summary_csv(results, output_dir / "summary.csv")
    write_slice_details_csv(results, output_dir / "slice_details.csv")
    write_html_report(results, output_dir / "report.html")

    total_patients = len(results)
    total_errors = sum(1 for result in results if result.status == "ERROR")
    print(f"Pacientes validados: {total_patients}")
    print(f"Pacientes com erro: {total_errors}")
    print(f"Relatorio HTML: {output_dir / 'report.html'}")
    print(f"Resumo CSV: {output_dir / 'summary.csv'}")
    print(f"Detalhes por fatia: {output_dir / 'slice_details.csv'}")


if __name__ == "__main__":
    main()
