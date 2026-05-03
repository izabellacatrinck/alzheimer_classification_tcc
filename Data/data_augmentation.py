"""
Pipeline Completo de Pré-processamento e Data Augmentation para MRI
Classificação Multiclasse (3 classes) com Desbalanceamento

Ordem das etapas:
1. Carregamento e validação dos dados
3. Resizing para dimensão padrão
4. Normalização de valores (0-1)
5. Data Augmentation com Albumentations
6. Split treino/validação/teste
7. Balanceamento via class weights
8. Salvamento dos dados processados
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MRIPreprocessor:
    """
    Classe para pré-processamento completo de imagens MRI
    """
    
    def __init__(self, input_dir, annotations_file, output_dir, img_size=(256, 256)):
        """
        Args:
            input_dir: diretório com arquivos .npy
            annotations_file: arquivo CSV/Excel com ID e diagnóstico
            output_dir: diretório para salvar dados processados
            img_size: tamanho padrão das imagens (altura, largura)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = img_size
        
        # Carregar anotações
        if str(annotations_file).endswith('.csv'):
            self.annotations = pd.read_csv(annotations_file)
        else:
            self.annotations = pd.read_excel(annotations_file)

        # 🔥 GARANTIA: apenas 1 visita por paciente
        if 'Acq Date' in self.annotations.columns:
            self.annotations['Acq Date'] = pd.to_datetime(
                self.annotations['Acq Date'],
                errors='coerce'
            )
            self.annotations = self.annotations.sort_values('Acq Date')

        self.annotations = self.annotations.drop_duplicates(
            subset='Subject',
            keep='first'
        )
        
        # Encoder para labels
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        self.augmentation_transform = None
        self.validation_transform = None
        
        print(f"✓ Anotações carregadas: {len(self.annotations)} pacientes")
        print(f"✓ Classes encontradas: {self.annotations['Group'].unique()}")
    
    # ========== ETAPA 2: RESIZING ==========
    def resize_image(self, img, size=None):
        """
        Redimensiona imagem mantendo proporção ou para tamanho fixo
        """
        if size is None:
            size = self.img_size
        
        from scipy.ndimage import zoom
        
        # Calcular fatores de zoom
        zoom_factors = (size[0] / img.shape[0], size[1] / img.shape[1])
        resized = zoom(img, zoom_factors, order=1)  # Interpolação linear
        
        return resized
    
    # ========== ETAPA 3: NORMALIZAÇÃO DE VALORES ==========
    def normalize_values(self, img):
        """
        Normaliza valores para [0, 1]
        """
        img_min = np.min(img)
        img_max = np.max(img)
        
        if img_max - img_min == 0:
            return np.zeros_like(img)
        
        return (img - img_min) / (img_max - img_min + 1e-8)
    
    # ========== ETAPA 4: CARREGAMENTO E PRÉ-PROCESSAMENTO ==========
    def load_and_preprocess(self, npy_file, verbose=False):
        """
        Carrega arquivo .npy e aplica todas as normalizações
        """
        try:
            img = np.load(npy_file)
            
            # Garantir que é 2D
            if len(img.shape) == 3:
                img = img[0] if img.shape[0] == 1 else img
            
            # Aplicar etapas de normalização
            img = self.resize_image(img)          # Resizing
            img = self.normalize_values(img)      # Min-max [0,1]
            
            if verbose:
                print(f"✓ {npy_file.name}: {img.shape}, min={img.min():.3f}, max={img.max():.3f}")
            
            return img
        
        except Exception as e:
            print(f"✗ Erro ao carregar {npy_file}: {e}")
            return None
    
    # ========== ETAPA 5: DATA AUGMENTATION ==========
    def setup_augmentation(self, augment_prob=0.8):
        """
        Define transformações de augmentation usando Albumentations
        IMPORTANTE: Aplicar APENAS no conjunto de treino
        """
        self.augmentation_transform = A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.2),  # depende da simetria
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.1),])
        
        # Transformação para validação/teste (apenas normalização)
        self.validation_transform = A.Compose([
            A.NoOp()
        ])
        
        print("✓ Augmentation configurada")
    
    def augment_image(self, img):
        """
        Aplica augmentation em uma imagem
        """
        if self.augmentation_transform is None:
            self.setup_augmentation()
        
        return self.augmentation_transform(image=img)['image']
    
     # ========== ETAPA 6: PROCESSAMENTO DO DATASET COMPLETO ==========
    def process_dataset(self):
        """
        Carrega, processa e organiza todo o dataset
        AGORA: apenas imagens da pasta axial e ID pelo diretório
        """
        print("\n" + "="*60)
        print("ETAPA 1: CARREGAMENTO E PRÉ-PROCESSAMENTO")
        print("="*60)
        
        images = []
        labels = []
        patient_ids = []
        
        # 🔥 ALTERAÇÃO 1: buscar apenas pasta axial
        npy_files = list(self.input_dir.glob("*/axial/*.npy"))
        
        if len(npy_files) == 0:
            print(f"✗ Nenhum arquivo .npy encontrado em {self.input_dir}")
            return None
        
        print(f"Processando {len(npy_files)} imagens...")
        valid_subjects = set(self.annotations['Subject'].astype(str))
        for npy_file in tqdm(npy_files):
            # 🔥 ALTERAÇÃO 2: pegar ID do diretório (não do nome do arquivo)
            patient_id = npy_file.parent.parent.name

            if patient_id not in valid_subjects:
                continue
            
            # Buscar label nas anotações
            matching = self.annotations[
                self.annotations['Subject'].astype(str) == str(patient_id)
            ]
            
            if len(matching) == 0:
                continue
            
            label = matching.iloc[0]['Group']
            
            img = self.load_and_preprocess(npy_file)
            
            if img is not None:
                images.append(img)
                labels.append(label)
                patient_ids.append(patient_id)
        
        if len(images) == 0:
            print("✗ Nenhuma imagem foi processada com sucesso")
            return None
        
        images = np.array(images)
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"\n✓ Dataset processado:")
        print(f"  - Total de amostras: {len(images)}")
        print(f"  - Shape de cada imagem: {images[0].shape}")
        print("\n🔎 Checagem final:")
        print("Pacientes únicos:", len(np.unique(patient_ids)))
        
        return {
            'images': images,
            'labels': labels_encoded,
            'labels_text': np.array(labels),
            'patient_ids': np.array(patient_ids)
        }
    
    # ========== ETAPA 7: ANÁLISE DE DESBALANCEAMENTO ==========
    def analyze_class_balance(self, labels):
        """
        Analisa e visualiza o balanceamento de classes
        """
        print("\n" + "="*60)
        print("ETAPA 2: ANÁLISE DE DESBALANCEAMENTO")
        print("="*60)
        
        unique, counts = np.unique(labels, return_counts=True)
        class_names = self.label_encoder.inverse_transform(unique)
        
        print("\nDistribuição de classes:")
        for name, count in zip(class_names, counts):
            percentage = (count / len(labels)) * 100
            print(f"  {name}: {count} ({percentage:.1f}%)")
        
        # Calcular class weights
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        print("\nClass weights (para balanceamento):")
        for name, weight in zip(class_names, self.class_weights):
            print(f"  {name}: {weight:.3f}")
        
        return unique, counts, class_names
    
    # ========== ETAPA 8: SPLIT TREINO/VALIDAÇÃO/TESTE ==========
    def split_dataset(self, data, test_size=0.2, val_size=0.1):
        """
        🔥 ALTERAÇÃO: SPLIT POR PACIENTE
        """
        print("\n" + "="*60)
        print("ETAPA 3: SPLIT TREINO/VALIDAÇÃO/TESTE (POR PACIENTE)")
        print("="*60)
        
        df = pd.DataFrame({
            'patient_id': data['patient_ids'],
            'label': data['labels']
        }).drop_duplicates()

        # split pacientes
        train_ids, test_ids = train_test_split(
            df['patient_id'],
            test_size=test_size,
            random_state=42,
            stratify=df['label']
        )

        val_size_adjusted = val_size / (1 - test_size)

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=df.set_index('patient_id').loc[train_ids]['label']
        )

        def filter_by_ids(ids):
            mask = np.isin(data['patient_ids'], ids)
            return (
                data['images'][mask],
                data['labels'][mask],
                data['patient_ids'][mask],
                data['labels_text'][mask]
            )

        X_train, y_train, ids_train, text_train = filter_by_ids(train_ids)
        X_val, y_val, ids_val, text_val = filter_by_ids(val_ids)
        X_test, y_test, ids_test, text_test = filter_by_ids(test_ids)

        print(f"\n✓ Dataset dividido (por paciente):")
        print(f"  - Treino: {len(train_ids)} pacientes")
        print(f"  - Validação: {len(val_ids)} pacientes")
        print(f"  - Teste: {len(test_ids)} pacientes")

        return {
            'X_train': X_train, 'y_train': y_train, 'ids_train': ids_train, 'text_train': text_train,
            'X_val': X_val, 'y_val': y_val, 'ids_val': ids_val, 'text_val': text_val,
            'X_test': X_test, 'y_test': y_test, 'ids_test': ids_test, 'text_test': text_test
        }
    
   # ========== ETAPA 9: APLICAÇÃO DE AUGMENTATION ==========
    def apply_augmentation_to_training(self, X_train, y_train, ids_train, augment_factor=2):

        self.setup_augmentation()

        X_augmented = list(X_train)
        y_augmented = list(y_train)
        ids_augmented = list(ids_train)  # 🔥 NOVO

        for _ in range(augment_factor - 1):
            for img, label, pid in tqdm(zip(X_train, y_train, ids_train), total=len(X_train)):
                augmented = self.augment_image(img)

                X_augmented.append(augmented)
                y_augmented.append(label)
                ids_augmented.append(pid)  # 🔥 DUPLICA ID

        return (
            np.array(X_augmented),
            np.array(y_augmented),
            np.array(ids_augmented)  # 🔥 RETORNA IDS
    )
    
   # ========== ETAPA 10: SALVAMENTO DOS DADOS ==========
    def save_processed_data(self, split_data, augmented=True):
        """
        🔥 ALTERAÇÃO: salvar também em pastas train/val/test
        """
        print("\n" + "="*60)
        print("ETAPA 5: SALVAMENTO DOS DADOS PROCESSADOS")
        print("="*60)
        
        if augmented:
            X_train_aug, y_train_aug, ids_train_aug = self.apply_augmentation_to_training(
            split_data['X_train'],
            split_data['y_train'],
            split_data['ids_train'],
            augment_factor=2
        )

            split_data['X_train'] = X_train_aug
            split_data['y_train'] = y_train_aug
            split_data['ids_train'] = ids_train_aug  
        
        for key in ['X_train', 'X_val', 'X_test']:
            split_data[key] = np.expand_dims(split_data[key], axis=-1)
        
        # 🔥 SALVAMENTO ORIGINAL (mantido)
        np.savez_compressed(
            self.output_dir / 'mri_processed.npz',
            X_train=split_data['X_train'],
            y_train=split_data['y_train'],
            X_val=split_data['X_val'],
            y_val=split_data['y_val'],
            X_test=split_data['X_test'],
            y_test=split_data['y_test']
        )

        # 🔥 ALTERAÇÃO: salvar em pastas
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)

            X = split_data[f'X_{split}']
            y = split_data[f'y_{split}']
            ids = split_data[f'ids_{split}']

            for i in range(len(X)):
                patient_dir = split_dir / str(ids[i])
                patient_dir.mkdir(exist_ok=True)
                np.save(patient_dir / f"img_{i}.npy", X[i])

        # metadados (inalterado)
        metadata = {
            'class_names': self.label_encoder.classes_.tolist(),
            'class_weights': self.class_weights.tolist(),
            'img_size': self.img_size,
            'n_classes': len(self.label_encoder.classes_),
        }

        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✓ Dados salvos em {self.output_dir}")
        
        return metadata
    
    # ========== VISUALIZAÇÃO ==========
    def visualize_samples(self, split_data, n_samples=3):
        """
        Visualiza amostras do dataset
        """
        fig, axes = plt.subplots(3, n_samples, figsize=(12, 10))
        
        class_names = self.label_encoder.classes_
        
        for i in range(3):
            if i == 0:
                X = split_data['X_train']
                y = split_data['y_train']
                title = "Treino"
            elif i == 1:
                X = split_data['X_val']
                y = split_data['y_val']
                title = "Validação"
            else:
                X = split_data['X_test']
                y = split_data['y_test']
                title = "Teste"
            
            for j in range(n_samples):
                idx = np.random.randint(0, len(X))
                img = X[idx] if X[idx].shape[-1] == 1 else X[idx]
                img = img.squeeze()
                
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].set_title(f"{title}\n{class_names[y[idx]]}")
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualização salva em sample_visualization.png")
        plt.close()

    def save_dataset_report(self, data, split_data):

        print("\nSalvando relatório do dataset...")

        # =============================
        # 📊 TAMANHOS
        # =============================
        total_images = len(data['images'])
        total_patients = len(np.unique(data['patient_ids']))

        train_images = len(split_data['X_train'])
        val_images = len(split_data['X_val'])
        test_images = len(split_data['X_test'])

        train_patients = len(np.unique(split_data['ids_train']))
        val_patients = len(np.unique(split_data['ids_val']))
        test_patients = len(np.unique(split_data['ids_test']))

        # =============================
        # 📊 DISTRIBUIÇÃO POR CLASSE (PACIENTE)
        # =============================
        df = pd.DataFrame({
            'patient_id': data['patient_ids'],
            'label': data['labels_text']
        }).drop_duplicates()

        class_counts = df['label'].value_counts()

        # =============================
        # 💾 SALVAR CSV
        # =============================
        report = pd.DataFrame({
            'metric': [
                'total_images', 'total_patients',
                'train_images', 'val_images', 'test_images',
                'train_patients', 'val_patients', 'test_patients'
            ],
            'value': [
                total_images, total_patients,
                train_images, val_images, test_images,
                train_patients, val_patients, test_patients
            ]
        })

        report.to_csv(self.output_dir / 'dataset_report.csv', index=False)

        class_counts.to_csv(self.output_dir / 'class_distribution.csv')

        # =============================
        # 📈 GRÁFICO
        # =============================
        plt.figure()
        class_counts.plot(kind='bar')
        plt.title("Distribuição de Pacientes por Classe")
        plt.xlabel("Classe")
        plt.ylabel("Número de Pacientes")

        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close()

        print("✓ Relatórios salvos em:", self.output_dir)


# ============================================================
# EXECUÇÃO DO PIPELINE
# ============================================================

def run_pipeline(input_dir, annotations_file, output_dir, img_size=(256, 256)):
    """
    Executa o pipeline completo
    """
    print("\n" + "="*60)
    print("PIPELINE DE PRÉ-PROCESSAMENTO DE MRI")
    print("="*60)
    
    # Inicializar
    preprocessor = MRIPreprocessor(
        input_dir=input_dir,
        annotations_file=annotations_file,
        output_dir=output_dir,
        img_size=img_size
    )
    
    # Processar dataset
    data = preprocessor.process_dataset()
    if data is None:
        return
    
    # Analisar balanceamento
    preprocessor.analyze_class_balance(data['labels'])
    
    # Split
    split_data = preprocessor.split_dataset(data)
    preprocessor.save_dataset_report(data, split_data)
    # Salvar com augmentation
    metadata = preprocessor.save_processed_data(split_data, augmented=True)
    # Visualizar
    preprocessor.visualize_samples(split_data, n_samples=3)
    
    print("\n" + "="*60)
    print("✓ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*60)
    
    return preprocessor, metadata


# ============================================================
# EXEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    # CONFIGURE ESSAS VARIÁVEIS COM SEUS DADOS:
    INPUT_DIR = "./ADNI_PROCESSADO"           # Diretório com arquivos .npy
    ANNOTATIONS_FILE = "./annotations.csv"  # Arquivo com ID e diagnóstico
    OUTPUT_DIR = "./ADNI_SPLITTED"       # Diretório para salvar
    IMG_SIZE = (256, 256)                # Tamanho padrão das imagens
    
    preprocessor, metadata = run_pipeline(
        input_dir=INPUT_DIR,
        annotations_file=ANNOTATIONS_FILE,
        output_dir=OUTPUT_DIR,
        img_size=IMG_SIZE
    )
    
    print("\nMetadados do dataset processado:")
    print(metadata)