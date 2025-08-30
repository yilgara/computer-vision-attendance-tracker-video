from embeddings import compute_embedding
from utils import cosine_similarity






def recognize_face_in_frame(frame, known_embeddings, known_names, known_ids, threshold=0.6):
    """Recognize faces in frame using pre-computed embeddings (MUCH FASTER!)"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Compute embedding for the frame using DeepFace in-memory
        embeddings_result = DeepFace.represent(
            img_path=None,  # Not using a path
            img=rgb_frame,  # Pass the NumPy array directly
            model_name='VGG-Face',
            enforce_detection=False
        )

        if not embeddings_result:
            return []

        # DeepFace.represent returns a list of dicts; get the embedding vector
        frame_embedding = embeddings_result[0]['embedding']
        
        recognized_names = []
        
        # Compare with all known embeddings using cosine similarity
        for i, known_embedding in enumerate(known_embeddings):
            similarity = cosine_similarity(frame_embedding, known_embedding)
            
            # If similarity is above threshold, it's a match
            if similarity > threshold:
                emp_id = known_ids[i]
                name = known_names[i]

                if not any(r['id'] == emp_id for r in recognized):
                    recognized.append({
                        'id': emp_id,
                        'name': name,
                        'similarity': similarity
                    })
                    st.write(f"🎯 Detected {name} (similarity: {similarity:.3f})")

        
        return recognized
    except Exception as e:
        st.warning(f"Face recognition error: {str(e)}")
        return []
