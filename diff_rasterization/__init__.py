

import torch

from . import cuda_rasterization




class rasterizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,colors,opacity,mean3w,q,s,camera):

        viewmatrix = camera.viewmatrix.cuda()
        projmatrix = camera.projmatrix.cuda()
        height     = camera.height
        width      = camera.width
        fx         = camera.fx
        fy         = camera.fy

        args = (colors,
                opacity,
                mean3w,
                q,      ##顺序依次为qr\qi\qj\qk
                s,
                viewmatrix,
                projmatrix,
                height,
                width,
                fx,
                fy)
        
        num_rendered,rendered_image, image_buffer,geometry_buffer,binning_buffer = cuda_rasterization.render(*args)
        ctx.num_rendered = num_rendered
        ctx.fx = fx
        ctx.fy = fy
        ctx.height = height
        ctx.width  = width
        ctx.save_for_backward(viewmatrix,projmatrix,image_buffer,binning_buffer,geometry_buffer,colors,opacity,mean3w,q,s)
        return rendered_image
        

    @staticmethod
    def backward(ctx, grad_outputs):
        viewmatrix,projmatrix,image_buffer,binning_buffer,geometry_buffer,colors,opacity,mean3w,q,s = ctx.saved_tensors
        args=(
            ctx.num_rendered,
            ctx.height,
            ctx.width,
            ctx.fx,
            ctx.fy,
            viewmatrix,
            projmatrix,
            image_buffer,
            binning_buffer,
            geometry_buffer,
            grad_outputs,
            colors,
            opacity,
            mean3w,
            q,
            s
        )

        dl_dcolor,dl_dopacity,dl_dq,dl_ds,dl_dcen3w = cuda_rasterization.backward(*args)
        return dl_dcolor,dl_dopacity,dl_dcen3w,dl_dq,dl_ds,None


