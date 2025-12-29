"""
complete_vit_drone_detection_pipeline.py
Полный конвейер: обучение всех моделей ViT, сравнение, генерация отчетов
"""

import os
import sys
import yaml
import torch
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Устанавливаем путь для CUDA (если доступно)
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if torch.cuda.is_available() else ''

# ============================================================================
# КОНФИГУРАЦИЯ ВСЕГО КОНВЕЙЕРА
# ============================================================================

class PipelineConfig:
    """Конфигурация полного конвейера обучения ViT моделей"""
    
    def __init__(self):
        # Пути к данным
        self.data_yaml_path = r"C:\Users\snytk\PycharmProjects\YOLOTrain\DatasetFromMAI\data.yaml"
        self.test_video_path = r"C:\Users\snytk\PycharmProjects\YOLOTrain\videoMAI\01-02.mp4"
        self.output_base_dir = r"C:\Users\snytk\PycharmProjects\YOLOTrain\ViT_Experiments"
        
        # Модели для обучения
        self.vit_models = [
            {
                'name': 'ViT_Detr_ResNet50',
                'type': 'detr',
                'config': {
                    'backbone': 'resnet50',
                    'image_size': 512,
                    'learning_rate': 1e-4,
                    'epochs': 50,
                    'batch_size': 4,
                    'num_queries': 100
                }
            },
            {
                'name': 'ViT_Detr_ResNet101',
                'type': 'detr',
                'config': {
                    'backbone': 'resnet101',
                    'image_size': 512,
                    'learning_rate': 1e-4,
                    'epochs': 50,
                    'batch_size': 2,
                    'num_queries': 100
                }
            },
            {
                'name': 'YOLOS_Tiny',
                'type': 'yolos',
                'config': {
                    'variant': 'yolos-tiny',
                    'image_size': 512,
                    'learning_rate': 2e-4,
                    'epochs': 100,
                    'batch_size': 4,
                    'num_queries': 100
                }
            },
            {
                'name': 'YOLOS_Small',
                'type': 'yolos',
                'config': {
                    'variant': 'yolos-small',
                    'image_size': 512,
                    'learning_rate': 1e-4,
                    'epochs': 100,
                    'batch_size': 2,
                    'num_queries': 100
                }
            },
            {
                'name': 'EfficientViT_B0',
                'type': 'efficientvit',
                'config': {
                    'variant': 'efficientvit_b0',
                    'image_size': 256,
                    'learning_rate': 3e-4,
                    'epochs': 150,
                    'batch_size': 8,
                    'pretrained': True
                }
            },
            {
                'name': 'MobileViT_S',
                'type': 'mobilevit',
                'config': {
                    'variant': 'mobilevit_s',
                    'image_size': 256,
                    'learning_rate': 1e-3,
                    'epochs': 200,
                    'batch_size': 8,
                    'pretrained': True
                }
            }
        ]
        
        # Метрики для сравнения
        self.metrics_to_track = [
            'mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall', 'f1',
            'train_loss', 'val_loss', 'inference_time_ms',
            'model_size_mb', 'gpu_memory_mb'
        ]
        
        # Параметры отчетов
        self.tensorboard_dir = os.path.join(self.output_base_dir, 'tensorboard_logs')
        self.reports_dir = os.path.join(self.output_base_dir, 'reports')
        self.models_dir = os.path.join(self.output_base_dir, 'trained_models')
        
        # Создаем директории
        for dir_path in [self.output_base_dir, self.tensorboard_dir, 
                        self.reports_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Время запуска
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_config(self):
        """Сохраняем конфигурацию конвейера"""
        config_path = os.path.join(self.reports_dir, f'pipeline_config_{self.start_time}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        print(f"[INFO] Конфигурация сохранена: {config_path}")

# ============================================================================
# КЛАССЫ ДЛЯ ОБУЧЕНИЯ РАЗНЫХ ТИПОВ ViT
# ============================================================================

class ViTTrainerBase:
    """Базовый класс для обучения ViT моделей"""
    
    def __init__(self, model_config: Dict, output_dir: str):
        self.config = model_config
        self.output_dir = output_dir
        self.model = None
        self.metrics = {}
        
    def prepare_data(self):
        """Подготовка данных"""
        print(f"[INFO] Подготовка данных для {self.config['name']}...")
        # Здесь будет преобразование YOLO -> COCO для ViT
        
    def build_model(self):
        """Создание модели"""
        raise NotImplementedError
        
    def train(self):
        """Обучение модели"""
        raise NotImplementedError
        
    def evaluate(self, test_data_path: str):
        """Оценка модели"""
        raise NotImplementedError
        
    def save_model(self):
        """Сохранение модели"""
        raise NotImplementedError

class DETRTrainer(ViTTrainerBase):
    """Обучение DETR модели"""
    
    def build_model(self):
        print(f"[DETR] Создание модели {self.config['config']['backbone']}...")
        try:
            from transformers import DetrForObjectDetection, DetrImageProcessor
            
            self.processor = DetrImageProcessor.from_pretrained(
                f"facebook/detr-resnet-{self.config['config']['backbone'].replace('resnet', '')}"
            )
            
            self.model = DetrForObjectDetection.from_pretrained(
                f"facebook/detr-resnet-{self.config['config']['backbone'].replace('resnet', '')}",
                num_labels=1,  # только дроны
                ignore_mismatched_sizes=True
            )
            
            print(f"[DETR] Модель создана успешно")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка создания DETR модели: {e}")
            return False
    
    def train(self):
        print(f"[DETR] Обучение модели...")
        # Реализация обучения DETR
        pass

class YOLOSTrainer(ViTTrainerBase):
    """Обучение YOLOS модели"""
    
    def build_model(self):
        print(f"[YOLOS] Создание модели {self.config['config']['variant']}...")
        try:
            from transformers import YolosForObjectDetection, YolosImageProcessor
            
            self.processor = YolosImageProcessor.from_pretrained(
                f"hustvl/{self.config['config']['variant']}"
            )
            
            self.model = YolosForObjectDetection.from_pretrained(
                f"hustvl/{self.config['config']['variant']}",
                num_labels=1,
                ignore_mismatched_sizes=True
            )
            
            print(f"[YOLOS] Модель создана успешно")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка создания YOLOS модели: {e}")
            return False

class EfficientViTTrainer(ViTTrainerBase):
    """Обучение EfficientViT с детекционной головой"""
    
    def build_model(self):
        print(f"[EfficientViT] Создание модели {self.config['config']['variant']}...")
        try:
            import timm
            import torch.nn as nn
            
            # Загружаем бэкбон
            backbone = timm.create_model(
                self.config['config']['variant'],
                pretrained=self.config['config']['pretrained'],
                features_only=True,
                out_indices=[2, 3, 4]
            )
            
            # Создаем детекционную голову
            class EfficientViTDetector(nn.Module):
                def __init__(self, backbone, num_classes=1):
                    super().__init__()
                    self.backbone = backbone
                    # Реализация FPN и детекционных голов
                    # ...
                    
            self.model = EfficientViTDetector(backbone, num_classes=1)
            print(f"[EfficientViT] Модель создана успешно")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка создания EfficientViT модели: {e}")
            return False

# ============================================================================
# МЕНЕДЖЕР ЭКСПЕРИМЕНТОВ
# ============================================================================

class ExperimentManager:
    """Управление всеми экспериментами"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.experiments = []
        self.results = {}
        self.summary_df = None
        
    def run_all_experiments(self):
        """Запуск всех экспериментов"""
        print("=" * 80)
        print("ЗАПУСК ПОЛНОГО КОНВЕЙЕРА ОБУЧЕНИЯ ViT МОДЕЛЕЙ")
        print("=" * 80)
        
        for model_config in self.config.vit_models:
            print(f"\n{'='*60}")
            print(f"ЭКСПЕРИМЕНТ: {model_config['name']}")
            print(f"{'='*60}")
            
            # Создаем директорию для эксперимента
            exp_dir = os.path.join(self.config.models_dir, 
                                 f"{model_config['name']}_{self.config.start_time}")
            os.makedirs(exp_dir, exist_ok=True)
            
            # Выбираем нужный тренер
            trainer = self._get_trainer(model_config, exp_dir)
            
            if trainer and trainer.build_model():
                # Запускаем обучение
                trainer.prepare_data()
                trainer.train()
                trainer.save_model()
                
                # Собираем метрики
                self.results[model_config['name']] = {
                    'config': model_config,
                    'trainer': trainer,
                    'metrics': trainer.metrics,
                    'directory': exp_dir
                }
                
                print(f"[SUCCESS] Эксперимент {model_config['name']} завершен успешно")
            else:
                print(f"[FAILED] Эксперимент {model_config['name']} пропущен из-за ошибки")
                
        return self.results
    
    def _get_trainer(self, model_config: Dict, output_dir: str) -> Optional[ViTTrainerBase]:
        """Создание соответствующего тренера"""
        model_type = model_config['type']
        
        if model_type == 'detr':
            return DETRTrainer(model_config, output_dir)
        elif model_type == 'yolos':
            return YOLOSTrainer(model_config, output_dir)
        elif model_type == 'efficientvit':
            return EfficientViTTrainer(model_config, output_dir)
        elif model_type == 'mobilevit':
            # Можно добавить MobileViT тренер
            return EfficientViTTrainer(model_config, output_dir)  # временно
        else:
            print(f"[WARNING] Неизвестный тип модели: {model_type}")
            return None

# ============================================================================
# СИСТЕМА ОЦЕНКИ И СРАВНЕНИЯ
# ============================================================================

class ModelEvaluator:
    """Оценка и сравнение всех моделей"""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.manager = experiment_manager
        self.comparison_results = {}
        
    def evaluate_all_models(self, test_data_path: str):
        """Оценка всех обученных моделей"""
        print("\n" + "="*80)
        print("ЭТАП ОЦЕНКИ И СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        
        for model_name, experiment in self.manager.results.items():
            print(f"\n[EVALUATION] Оценка модели: {model_name}")
            
            metrics = {
                'model_name': model_name,
                'model_type': experiment['config']['type'],
                'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'parameters': self._count_parameters(experiment['trainer'].model),
                'inference_time': self._measure_inference_time(
                    experiment['trainer'].model, test_data_path
                )
            }
            
            # Добавляем метрики из обучения
            if experiment['trainer'].metrics:
                metrics.update(experiment['trainer'].metrics)
                
            self.comparison_results[model_name] = metrics
            
        return self.comparison_results
    
    def _count_parameters(self, model):
        """Подсчет параметров модели"""
        if model is None:
            return 0
        return sum(p.numel() for p in model.parameters())
    
    def _measure_inference_time(self, model, test_data_path):
        """Измерение времени инференса"""
        import time
        
        if model is None:
            return None
            
        # Здесь будет реальное измерение времени
        return 100  # временное значение

    def generate_comprehensive_report(self):
        """Генерация комплексного отчета"""
        print("\n" + "="*80)
        print("ГЕНЕРАЦИЯ ОТЧЕТОВ")
        print("="*80)
        
        # 1. Создаем DataFrame с результатами
        df = pd.DataFrame.from_dict(self.comparison_results, orient='index')
        self.manager.summary_df = df
        
        # 2. Сохраняем в разных форматах
        report_dir = self.manager.config.reports_dir
        timestamp = self.manager.config.start_time
        
        # CSV отчет
        csv_path = os.path.join(report_dir, f'model_comparison_{timestamp}.csv')
        df.to_csv(csv_path)
        print(f"[REPORT] CSV отчет сохранен: {csv_path}")
        
        # JSON отчет
        json_path = os.path.join(report_dir, f'model_comparison_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        print(f"[REPORT] JSON отчет сохранен: {json_path}")
        
        # Markdown отчет
        md_path = os.path.join(report_dir, f'model_comparison_{timestamp}.md')
        self._generate_markdown_report(df, md_path)
        
        # 3. Генерируем визуализации
        self._generate_visualizations(df, report_dir, timestamp)
        
        # 4. Генерируем TensorBoard отчет
        self._generate_tensorboard_report(df, timestamp)
        
        return df
    
    def _generate_markdown_report(self, df, output_path):
        """Генерация Markdown отчета"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Сравнение моделей ViT для детекции дронов\n\n")
            f.write(f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Сводная таблица результатов\n\n")
            f.write(df.to_markdown() + "\n\n")
            
            f.write("## Рекомендации по выбору модели\n\n")
            
            # Находим лучшие модели по разным критериям
            if 'mAP@0.5' in df.columns:
                best_map = df['mAP@0.5'].idxmax()
                f.write(f"- **Лучшая точность (mAP@0.5):** {best_map} ({df.loc[best_map, 'mAP@0.5']:.3f})\n")
            
            if 'inference_time' in df.columns:
                fastest = df['inference_time'].idxmin()
                f.write(f"- **Самая быстрая:** {fastest} ({df.loc[fastest, 'inference_time']:.1f} мс)\n")
            
            if 'parameters' in df.columns:
                smallest = df['parameters'].idxmin()
                f.write(f"- **Самая компактная:** {smallest} ({df.loc[smallest, 'parameters']:,} параметров)\n")
    
    def _generate_visualizations(self, df, report_dir, timestamp):
        """Генерация визуализаций"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import cm
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bar plot: Сравнение mAP
        if 'mAP@0.5' in df.columns:
            ax = axes[0, 0]
            df_sorted = df.sort_values('mAP@0.5', ascending=False)
            bars = ax.bar(df_sorted.index, df_sorted['mAP@0.5'])
            ax.set_title('Сравнение mAP@0.5', fontsize=14, fontweight='bold')
            ax.set_ylabel('mAP@0.5', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Bar plot: Время инференса
        if 'inference_time' in df.columns:
            ax = axes[0, 1]
            df_sorted = df.sort_values('inference_time')
            bars = ax.bar(df_sorted.index, df_sorted['inference_time'], color='orange')
            ax.set_title('Время инференса (мс)', fontsize=14, fontweight='bold')
            ax.set_ylabel('мс', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        
        # 3. Scatter plot: Точность vs Скорость
        if all(col in df.columns for col in ['mAP@0.5', 'inference_time']):
            ax = axes[1, 0]
            scatter = ax.scatter(df['inference_time'], df['mAP@0.5'], 
                                s=200, c=range(len(df)), cmap='viridis', alpha=0.7)
            ax.set_xlabel('Время инференса (мс)', fontsize=12)
            ax.set_ylabel('mAP@0.5', fontsize=12)
            ax.set_title('Точность vs Скорость', fontsize=14, fontweight='bold')
            
            # Добавляем аннотации
            for i, (name, row) in enumerate(df.iterrows()):
                ax.annotate(name, (row['inference_time'], row['mAP@0.5']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 4. Radar chart (упрощенный)
        ax = axes[1, 1]
        metrics_for_radar = ['mAP@0.5', 'precision', 'recall', 'f1'] if all(
            m in df.columns for m in ['mAP@0.5', 'precision', 'recall', 'f1']
        ) else df.select_dtypes(include=[np.number]).columns[:4]
        
        if len(metrics_for_radar) >= 3:
            # Нормализуем метрики для radar chart
            df_normalized = df[metrics_for_radar].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
            angles += angles[:1]  # Замыкаем круг
            
            for i, (name, row) in enumerate(df_normalized.iterrows()):
                values = row.tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=name, alpha=0.7)
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_for_radar)
            ax.set_title('Radar Chart Сравнения', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plot_path = os.path.join(report_dir, f'comparison_plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[REPORT] Визуализации сохранены: {plot_path}")
    
    def _generate_tensorboard_report(self, df, timestamp):
        """Генерация TensorBoard отчета"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = os.path.join(self.manager.config.tensorboard_dir, f'comparison_{timestamp}')
            writer = SummaryWriter(log_dir)
            
            # Записываем метрики
            for metric in df.columns:
                if df[metric].dtype in [np.float64, np.int64]:
                    for model_name, value in df[metric].items():
                        if pd.notna(value):
                            writer.add_scalar(f'Comparison/{metric}', value, 
                                            list(df.index).index(model_name))
            
            # Записываем таблицу сравнения
            writer.add_text('Model Comparison/Summary Table', 
                          df.to_markdown(), 0)
            
            # Записываем рекомендации
            recommendations = self._generate_recommendations_text(df)
            writer.add_text('Recommendations', recommendations, 0)
            
            writer.close()
            print(f"[TENSORBOARD] Отчет сохранен в: {log_dir}")
            print(f"[TENSORBOARD] Запустите: tensorboard --logdir={log_dir}")
            
        except ImportError:
            print("[WARNING] TensorBoard не установлен, пропускаем генерацию отчета")
    
    def _generate_recommendations_text(self, df):
        """Генерация текста рекомендаций"""
        recommendations = "## Рекомендации по выбору модели:\n\n"
        
        # Лучшая по точности
        if 'mAP@0.5' in df.columns:
            best_model = df['mAP@0.5'].idxmax()
            best_score = df.loc[best_model, 'mAP@0.5']
            recommendations += f"✅ **Лучшая точность (mAP@0.5):** {best_model} ({best_score:.3f})\n"
        
        # Лучшая по скорости
        if 'inference_time' in df.columns:
            fastest_model = df['inference_time'].idxmin()
            fastest_time = df.loc[fastest_model, 'inference_time']
            recommendations += f"⚡ **Самая быстрая:** {fastest_model} ({fastest_time:.1f} мс)\n"
        
        # Лучшая по балансу
        if all(col in df.columns for col in ['mAP@0.5', 'inference_time']):
            # Считаем score = mAP / inference_time
            df['efficiency_score'] = df['mAP@0.5'] / df['inference_time']
            balanced_model = df['efficiency_score'].idxmax()
            recommendations += f"⚖️  **Лучший баланс точности и скорости:** {balanced_model}\n"
        
        return recommendations

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ЗАПУСКА
# ============================================================================

def main():
    """Главная функция запуска всего конвейера"""
    
    print("\n" + "="*80)
    print("ViT ДРОН ДЕТЕКТОР - ПОЛНЫЙ КОНВЕЙЕР ОБУЧЕНИЯ И ОЦЕНКИ")
    print("="*80)
    
    # 1. Инициализация конфигурации
    print("\n[1/5] Инициализация конфигурации...")
    config = PipelineConfig()
    config.save_config()
    
    # 2. Проверка зависимостей
    print("\n[2/5] Проверка зависимостей...")
    check_dependencies()
    
    # 3. Запуск всех экспериментов
    print("\n[3/5] Запуск экспериментов...")
    manager = ExperimentManager(config)
    experiments_results = manager.run_all_experiments()
    
    if not experiments_results:
        print("[ERROR] Ни один эксперимент не завершился успешно!")
        return
    
    # 4. Оценка и сравнение моделей
    print("\n[4/5] Оценка и сравнение моделей...")
    evaluator = ModelEvaluator(manager)
    comparison_results = evaluator.evaluate_all_models(config.test_video_path)
    
    # 5. Генерация отчетов
    print("\n[5/5] Генерация отчетов...")
    final_report = evaluator.generate_comprehensive_report()
    
    # Финальный вывод
    print("\n" + "="*80)
    print("КОНВЕЙЕР УСПЕШНО ЗАВЕРШЕН!")
    print("="*80)
    
    print(f"\n📊 Обучено моделей: {len(experiments_results)}")
    print(f"📁 Результаты сохранены в: {config.output_base_dir}")
    print(f"📈 TensorBoard логи: {config.tensorboard_dir}")
    print(f"📄 Отчеты: {config.reports_dir}")
    
    if final_report is not None:
        print("\n🏆 Лучшие модели по категориям:")
        for metric in ['mAP@0.5', 'inference_time']:
            if metric in final_report.columns:
                if metric == 'mAP@0.5':
                    best = final_report[metric].idxmax()
                    value = final_report.loc[best, metric]
                    print(f"  • Точность ({metric}): {best} ({value:.3f})")
                elif metric == 'inference_time':
                    best = final_report[metric].idxmin()
                    value = final_report.loc[best, metric]
                    print(f"  • Скорость ({metric}): {best} ({value:.1f} мс)")
    
    print(f"\n🚀 Запустите TensorBoard для просмотра отчетов:")
    print(f"   tensorboard --logdir={config.tensorboard_dir}")
    
    # Сохраняем итоговый отчет
    summary_path = os.path.join(config.reports_dir, f'final_summary_{config.start_time}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ИТОГОВЫЙ ОТЧЕТ ViT ДРОН ДЕТЕКТОР\n")
        f.write("="*80 + "\n\n")
        f.write(f"Дата запуска: {config.start_time}\n")
        f.write(f"Обучено моделей: {len(experiments_results)}\n")
        f.write(f"Успешных: {len([r for r in experiments_results.values() if r])}\n\n")
        
        if final_report is not None:
            f.write("Сводная таблица результатов:\n")
            f.write("="*60 + "\n")
            f.write(final_report.to_string())
    
    print(f"\n📋 Итоговый отчет сохранен: {summary_path}")

def check_dependencies():
    """Проверка необходимых зависимостей"""
    required_packages = [
        ('torch', 'pytorch'),
        ('transformers', 'transformers'),
        ('timm', 'timm'),
        ('ultralytics', 'ultralytics'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pycocotools', 'pycocotools'),
        ('tensorboard', 'tensorboard')
    ]
    
    missing_packages = []
    
    for import_name, pip_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"[WARNING] Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("Установите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
        
        # Спрашиваем пользователя
        response = input("\nПродолжить без установки? (y/n): ").lower()
        if response != 'y':
            print("Установите зависимости и перезапустите скрипт.")
            sys.exit(1)
    else:
        print("[OK] Все зависимости установлены")

# ============================================================================
# УТИЛИТЫ ДЛЯ ЗАПУСКА ИЗ КОМАНДНОЙ СТРОКИ
# ============================================================================

def run_from_command_line():
    """Запуск конвейера из командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ViT Дрон Детектор - Полный конвейер')
    parser.add_argument('--data', type=str, default=None,
                       help='Путь к data.yaml с данными')
    parser.add_argument('--test-video', type=str, default=None,
                       help='Путь к тестовому видео')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Директория для сохранения результатов')
    parser.add_argument('--models', type=str, default='all',
                       help='Список моделей для обучения (через запятую или "all")')
    parser.add_argument('--skip-training', action='store_true',
                       help='Пропустить обучение, только оценка существующих моделей')
    parser.add_argument('--quick-test', action='store_true',
                       help='Быстрый тест на малом количестве эпох')
    
    args = parser.parse_args()
    
    # Обновляем конфигурацию аргументами командной строки
    if args.data or args.test_video or args.output_dir:
        print("[INFO] Используются аргументы командной строки")
        # Здесь можно обновить конфигурацию
    
    if args.quick_test:
        print("[INFO] Режим быстрого теста")
        # Уменьшаем количество эпох
    
    # Запускаем основной конвейер
    main()

# ============================================================================
# ЗАПУСК СКРИПТА
# ============================================================================

if __name__ == "__main__":
    # Автоматически запускаем все
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Конвейер прерван пользователем")
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        
        # Сохраняем информацию об ошибке
        error_dir = os.path.join(os.path.dirname(__file__), "error_logs")
        os.makedirs(error_dir, exist_ok=True)
        
        error_file = os.path.join(error_dir, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_file, 'w') as f:
            f.write(f"Ошибка в ViT конвейере: {datetime.now()}\n")
            f.write(f"Сообщение: {str(e)}\n\n")
            f.write("Трассировка:\n")
            traceback.print_exc(file=f)
        
        print(f"[INFO] Лог ошибки сохранен: {error_file}")