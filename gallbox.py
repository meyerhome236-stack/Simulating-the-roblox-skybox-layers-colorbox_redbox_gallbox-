#!/usr/bin/env python3
import math, random, pygame, pygame.gfxdraw

WIDTH, HEIGHT = 1280, 720
FPS = 60

def clamp(x, a, b): return a if x < a else b if x > b else x
def lerp(a, b, t): return a + (b - a) * t
def fade(t): return t * t * (3 - 2 * t)

class Camera:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.pos = [0.0, 0.0, 0.0]
        self.speed = 1.0
    def forward_flat(self):
        return (math.sin(self.yaw), 0.0, math.cos(self.yaw))
    def forward_vec(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return (sy*cp, -sp, cy*cp)
    def right(self):
        return (math.cos(self.yaw), 0.0, -math.sin(self.yaw))
    def apply(self, v):
        x,y,z = v[0]-self.pos[0], v[1]-self.pos[1], v[2]-self.pos[2]
        cy, sy = math.cos(-self.yaw), math.sin(-self.yaw)
        x, z = cy*x - sy*z, sy*x + cy*z
        cp, sp = math.cos(-self.pitch), math.sin(-self.pitch)
        y, z = cp*y - sp*z, sp*y + cp*z
        return (x, y, z)

def project(v):
    x,y,z = v
    z = (z + 3.0)
    if z <= 0.1: z = 0.1
    f = 780 / z
    return (int(WIDTH/2 + x*f), int(HEIGHT/2 - y*f))

# ---------- Noise ----------
class SmoothNoise:
    def __init__(self, seed=0, period=64):
        random.seed(seed)
        self.period = period
        self.grid = [[random.random() for _ in range(period+1)] for _ in range(period+1)]
    def sample(self, x, y):
        p = self.period
        x0 = int(math.floor(x)) % p; x1 = (x0 + 1) % p
        y0 = int(math.floor(y)) % p; y1 = (y0 + 1) % p
        fx = x - math.floor(x); fy = y - math.floor(y)
        v00 = self.grid[y0][x0]
        v10 = self.grid[y0][x1]
        v01 = self.grid[y1][x0]
        v11 = self.grid[y1][x1]
        sx = fade(fx); sy = fade(fy)
        ix0 = lerp(v00, v10, sx)
        ix1 = lerp(v01, v11, sx)
        return lerp(ix0, ix1, sy)

def green_palette(v):
    g = int(lerp(170, 240, v))
    r = int(lerp(120, 165, 0.8*v))
    b = int(lerp(150, 195, 0.5 + 0.5*v))
    return (r, g, b)


class FluidBG:
    def __init__(self, tiles=24, period=96, speed=0.16, drift=0.05):
        self.tiles = tiles
        self.n_color = SmoothNoise(seed=2222, period=period)
        self.n_flowx = SmoothNoise(seed=3333, period=period)
        self.n_flowy = SmoothNoise(seed=4444, period=period)
        self.speed = speed
        self.drift = drift
        self.ox = 0.0; self.oy = 0.0; self.tcolor = 0.0
    def step(self, dt):
        self.ox += self.speed * dt
        self.oy += self.speed * 0.77 * dt
        self.tcolor += self.drift * dt
    def draw(self, screen, t):
        w, h = screen.get_size()
        cols = self.tiles
        rows = int(self.tiles * (h / w)); rows = max(rows, 12)
        tw = w / cols; th = h / rows
        for j in range(rows):
            for i in range(cols):
                x0 = i * tw; y0 = j * th
                x1 = (i+1) * tw; y1 = (j+1) * th
                cx = (x0 + x1) * 0.5; cy = (y0 + y1) * 0.5
                nx = (cx / w) * 3.0 + self.ox
                ny = (cy / h) * 3.0 + self.oy
                v = self.n_color.sample(nx + self.tcolor*0.7, ny - self.tcolor*0.6)
                col = green_palette(v)
                fx = (self.n_flowx.sample(nx + t*0.22, ny) - 0.5) * tw * 0.8
                fy = (self.n_flowy.sample(nx, ny + t*0.22) - 0.5) * th * 0.8
                p0 = (x0 + fx*0.28, y0 + fy*0.28)
                p1 = (x1 + fx*0.22, y0 + fy*0.22)
                p2 = (x1 - fx*0.28, y1 - fy*0.28)
                p3 = (x0 - fx*0.22, y1 - fy*0.22)
                pygame.draw.polygon(screen, col, [p0,p1,p2,p3], 0)


class Phantom:
    FLOW = SmoothNoise(seed=8088, period=256)
    def __init__(self, bounds=10.0):
        self.bounds = bounds
        self.nverts = random.randint(3,4)  
        self.size = random.uniform(0.08, 0.16)  
        self.seed = random.random()*1000.0
        self.pos = [random.uniform(-bounds, bounds),
                    random.uniform(-bounds*0.6, bounds*0.6),
                    random.uniform(-bounds, bounds)]
        self.vel = [random.uniform(-0.12,0.12), random.uniform(-0.10,0.10), random.uniform(-0.12,0.12)]
        self.rot = random.random()*math.tau
     
        self.target = None; self.form_t = 0.0; self.form_timer = 0.0; self.is_forming = False

    def start_form(self, target_pos):
        tx,ty,tz = target_pos
       
        self.pos = [tx, ty, tz]
        self.vel = [0.0,0.0,0.0]
        self.target = [tx,ty,tz]; self.is_forming = True; self.form_t = 0.0; self.form_timer = 0.0

    def release_form(self):
        self.is_forming = False; self.target = None

    def update(self, dt, t):
        if self.is_forming and self.target is not None:
            self.pos[0], self.pos[1], self.pos[2] = self.target
            self.form_t = min(1.0, self.form_t + dt*2.5)
            self.form_timer += dt
            if self.form_timer > 2.5:
                self.release_form()
        else:
            fx = Phantom.FLOW.sample(self.pos[0]*0.3 + t*0.2, self.pos[2]*0.3) - 0.5
            fz = Phantom.FLOW.sample(self.pos[2]*0.3 - t*0.2, self.pos[0]*0.3) - 0.5
            fy = 0.25*math.sin(t*0.6 + self.seed)
            self.vel[0] += fx * 0.8 * dt
            self.vel[2] += fz * 0.8 * dt
            self.vel[1] += fy * 0.10 * dt
            d = 0.5
            self.vel[0] *= (1.0 - d*dt)
            self.vel[1] *= (1.0 - d*dt)
            self.vel[2] *= (1.0 - d*dt)
            self.pos[0] += self.vel[0] * dt
            self.pos[1] += self.vel[1] * dt
            self.pos[2] += self.vel[2] * dt
            b = self.bounds
            for i,lim in enumerate((b, b*0.7, b)):
                if self.pos[i] < -lim: self.pos[i] += 2*lim
                elif self.pos[i] > lim: self.pos[i] -= 2*lim
            self.form_t = max(0.0, self.form_t - dt*0.7)

      
        self.rot += (1.2 + 0.6*math.sin(t*1.3 + self.seed)) * dt

    def draw(self, screen, cam, t):
       
        alpha = int(clamp((0.25 + 0.65*self.form_t) * 255, 35, 220))
        size = self.size * (0.9 + 0.2*self.form_t) 
        verts2d = []
        ang0 = self.rot
        for k in range(self.nverts):
            ang = ang0 + k * (2*math.pi/self.nverts)
            r = size * (1.0 + 0.1*math.sin(t*8 + k*1.7 + self.seed))
            vx = self.pos[0] + math.cos(ang)*r
            vy = self.pos[1] + math.sin(ang)*r
            vz = self.pos[2]
            verts2d.append(project(cam.apply((vx,vy,vz))))
        pygame.gfxdraw.filled_polygon(screen, verts2d, (120, 220, 190, alpha))
        pygame.gfxdraw.aapolygon(screen, verts2d, (120, 220, 190, alpha))


def make_text_outline_targets(text, cam, distance=8.0, world_scale=0.010, max_points=200):
    font = pygame.font.SysFont("consolas,monospace", 220, bold=True)
    surf = font.render(text, True, (255,255,255))
    w, h = surf.get_size()
    alpha = pygame.surfarray.array_alpha(surf)
    edges = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if alpha[x][y] > 10:
                if alpha[x-1][y] <= 10 or alpha[x+1][y] <= 10 or alpha[x][y-1] <= 10 or alpha[x][y+1] <= 10:
                    edges.append((x,y))
 
    stride = max(1, int(max(w,h)/220))
    edge2 = edges[::stride]
    random.shuffle(edge2)
    if len(edge2) > max_points:
        edge2 = edge2[:max_points]

    fx, fy, fz = cam.forward_vec()
    up = (0.0, 1.0, 0.0)
    rx = up[1]*fz - up[2]*fy
    ry = up[2]*fx - up[0]*fz
    rz = up[0]*fy - up[1]*fx
    rl = max(1e-6, (rx*rx + ry*ry + rz*rz) ** 0.5)
    rx, ry, rz = rx/rl, ry/rl, rz/rl
    ux = fy*rz - fz*ry
    uy = fz*rx - fx*rz
    uz = fx*ry - fy*rx
    cx = cam.pos[0] + fx*distance
    cy = cam.pos[1] + fy*distance
    cz = cam.pos[2] + fz*distance

    pts = []
    for (x,y) in edge2:
        sx = (x - w/2) * world_scale
        sy = (y - h/2) * world_scale
        px = cx + rx * sx + ux * (-sy)
        py = cy + ry * sx + uy * (-sy)
        pz = cz + rz * sx + uz * (-sy)
        pts.append((px,py,pz))
    return pts

def main():
    pygame.init()
    pygame.display.set_caption("gallbox they say hi!")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    cam = Camera()
    mouse_locked = False
    invert_pitch = False

    fluid = FluidBG(tiles=24, period=96, speed=0.16, drift=0.05)
    phantoms = [Phantom(bounds=10.0) for _ in range(120)]  

    running = True
    while True:
        dt = clock.tick(FPS) / 1000.0
        t = pygame.time.get_ticks() / 1000.0
        wheel_move = 0.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT: return
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: return
                elif e.key == pygame.K_l:
                    mouse_locked = not mouse_locked
                    pygame.mouse.set_visible(not mouse_locked)
                    pygame.event.set_grab(mouse_locked)
                elif e.key == pygame.K_i:
                    invert_pitch = not invert_pitch
                elif e.key == pygame.K_z:
                    cam.pos = [0.0,0.0,0.0]; cam.yaw = 0.0; cam.pitch = 0.0
                elif e.key == pygame.K_v:
                    targets = make_text_outline_targets("HI", cam, distance=8.0, world_scale=0.010, max_points= min(200, len(phantoms)))
                    
                    ph_sorted = sorted(phantoms, key=lambda p: (p.pos[0]-cam.pos[0])**2 + (p.pos[1]-cam.pos[1])**2 + (p.pos[2]-cam.pos[2])**2)
                    take = min(len(targets), len(ph_sorted))
                    for i in range(take):
                        ph_sorted[i].start_form(targets[i])
            elif e.type == pygame.MOUSEWHEEL:
                wheel_move += e.y

        if mouse_locked:
            mx, my = pygame.mouse.get_rel()
            cam.yaw += mx * 0.0030
            cam.pitch = max(-1.2, min(1.2, cam.pitch + (my * 0.0030 if invert_pitch else -my * 0.0030)))
        else:
            pygame.mouse.get_rel()

        keys = pygame.key.get_pressed()
        boost = 2.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0
        spd = cam.speed * dt * boost
        fx, _, fz = cam.forward_flat()
        rx, _, rz = cam.right()
        if keys[pygame.K_w]: cam.pos[0] += fx*spd; cam.pos[2] += fz*spd
        if keys[pygame.K_s]: cam.pos[0] -= fx*spd; cam.pos[2] -= fz*spd
        if keys[pygame.K_a]: cam.pos[0] -= rx*spd; cam.pos[2] -= rz*spd
        if keys[pygame.K_d]: cam.pos[0] += rx*spd; cam.pos[2] += rz*spd
        if keys[pygame.K_q]: cam.pos[1] -= spd
        if keys[pygame.K_e]: cam.pos[1] += spd

        if wheel_move != 0.0:
            mx, my = pygame.mouse.get_pos()
            f = 780.0
            rx_cam = (mx - WIDTH/2) / f
            ry_cam = -(my - HEIGHT/2) / f
            rz_cam = 1.0
            cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
            cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
            x1, y1, z1 = rx_cam, ry_cam*cp + rz_cam*sp, -ry_cam*sp + rz_cam*cp
            dx, dy, dz = x1*cy - z1*sy, y1, x1*sy + z1*cy
            L = (dx*dx + dy*dy + dz*dz) ** 0.5 or 1.0
            dx, dy, dz = dx/L, dy/L, dz/L
            wheel_speed = 1.5 * wheel_move * (1.0 + 0.5*boost)
            cam.pos[0] += dx * wheel_speed * dt * 60.0
            cam.pos[1] += dy * wheel_speed * dt * 60.0
            cam.pos[2] += dz * wheel_speed * dt * 60.0

        fluid.step(dt)
        for p in phantoms:
            p.update(dt, t)

        fluid.draw(screen, t)
        for p in phantoms:
            p.draw(screen, cam, t)

        # HUD
        font = pygame.font.SysFont("consolas,monospace", 16)
        hud = f"[gallbox — V: says HI (polygon dudes)] [FPS {clock.get_fps():.1f}]"
        help1 = "L lock | Wheel move-to-cursor | WASD+QE | Shift | V form 'HI' | I invert | Z reset | ESC quit"
        screen.blit(font.render(hud, True, (15,30,15)), (10, 8))
        screen.blit(font.render(help1, True, (20,50,20)), (10, 28))

        pygame.display.flip()

if __name__ == '__main__':
    main()
