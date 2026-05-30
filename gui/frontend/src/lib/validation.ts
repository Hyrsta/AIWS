export interface Dim { w: number; h: number; }
export const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
export function sameDimensions(a: Dim, b: Dim): boolean { return a.w === b.w && a.h === b.h; }
export function readImageDim(file: File): Promise<Dim> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve({ w: img.naturalWidth, h: img.naturalHeight }); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("decode failed")); };
    img.src = url;
  });
}
