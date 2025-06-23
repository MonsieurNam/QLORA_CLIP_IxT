# %%writefile utils.py
import torch

def clip_classifier_original(classnames, template, clip_model):
    """
    Tạo ra các vector trọng số cho bộ phân loại từ text prompts.
    Phiên bản này dùng thư viện 'clip' gốc.
    """
    import clip # Cần import clip ở đây
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights


def pre_load_features_original(clip_model, loader):
    """
    Tối ưu hóa bằng cách mã hóa trước toàn bộ đặc trưng ảnh.
    Phiên bản này dùng thư viện 'clip' gốc.
    """
    from tqdm import tqdm # Cần import tqdm ở đây
    
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
            
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels