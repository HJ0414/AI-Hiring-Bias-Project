from __future__ import annotations
import hashlib, os, requests, zipfile, shutil, logging
from pathlib import Path
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.data_dir.mkdir(exist_ok=True, parents=True)
    
    # def _download(self, url: str, dest: Path, sha256: str | None = None):
    #     if dest.exists():
    #         if sha256 and self._sha(dest) != sha256:
    #             logger.warning("%s checksum mismatch, redownloading", dest.name)
    #             dest.unlink()  # 修复
    #         else:
    #             return dest
    #     
    #     logger.info("Downloading %s → %s", url, dest)
    #     with requests.get(url, stream=True, timeout=60) as r:
    #         r.raise_for_status()
    #         total = int(r.headers.get('content-length', 0))
    #         with open(dest, 'wb') as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
    #             for chunk in r.iter_content(chunk_size=8192):
    #                 f.write(chunk)
    #                 pbar.update(len(chunk))
    #     
    #     if sha256 and self._sha(dest) != sha256:  # 修复：加上(dest)
    #         raise ValueError(f"Checksum failed for {dest}")
    #     return dest
    
    @staticmethod
    def _sha(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    
    def fetch_all(self):
        if not (self.cfg.data_dir / 'bios_bias.csv').exists():
            logger.info("Creating sample BIOS dataset...")
            self._create_sample_bios_data()

        logger.info("Dataset ready")
        return self.cfg.data_dir
    
    def _create_sample_bios_data(self):
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        n_samples = 50000

        tech_jobs = ['software engineer', 'data scientist', 'system administrator', 'security analyst']
        care_jobs = ['nurse', 'teacher', 'social worker', 'therapist']
        business_jobs = ['accountant', 'manager', 'consultant', 'lawyer']
        
        all_jobs = tech_jobs + care_jobs + business_jobs

        data = []
        for i in range(n_samples):
            job = np.random.choice(all_jobs)

            if job in tech_jobs:
                gender = np.random.choice([0,1], p=[0.3, 0.7])
            elif job in care_jobs:
                gender = np.random.choice([0,1], p= [0.8, 0.2])
            else:
                gender = np.random.choice([0,1], p = [0.5, 0.5])
            
            skills = np.random.choice([
                'strong communication skills', 'team leadership', 'problem solving',
                'analytical thinking', 'project management', 'customer service'
            ], size = 2, replace = False)

            bio = f"I am a {job} with experience in {',' .join(skills)}."+\
                  f"I have {np.random.randint(1, 15)} years of professional experience."
            
            data.append({
                'bio' : bio,
                'gender' : gender,
                'label': all_jobs.index(job),
                'profession': job
            })

        df = pd.DataFrame(data)
        df.to_csv(self.cfg.data_dir/'bios_bias.csv', index = False)
        logger.info(f"Created Bios Dataset, {df['gender'].value_counts().to_dict()}")
        

    
            
        
        
            


        
        
        
        
    
    


